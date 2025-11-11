import carla
import numpy as np 
import os
import math
import cv2

def left_right_boundaries_from_centerline(center_pts):
    left_pts = []
    right_pts = []
    c_pts = []
    for wp in center_pts:
        loc = wp.transform.location
        yaw_deg = wp.transform.rotation.yaw
        yaw_rad = math.radians(yaw_deg)

        # 차선 진행 방향 단위 벡터
        dx = math.cos(yaw_rad)
        dy = math.sin(yaw_rad)

        # 왼쪽(법선) 단위 벡터 = 회전(+90도): (-dy, dx)
        nx = -dy
        ny = dx

        half_w = wp.lane_width * 0.5

        # 왼 경계
        lx = loc.x + nx * half_w
        ly = loc.y + ny * half_w
        lz = loc.z
        left_pts.append((lx, ly, lz))

        # 오른 경계 (반대 방향)
        rx = loc.x - nx * half_w
        ry = loc.y - ny * half_w
        rz = loc.z
        right_pts.append((rx, ry, rz))

        c_pts.append((loc.x, loc.y, loc.z))

    return left_pts, right_pts, c_pts

def get_lane_info(world_map, cur_wp, dist_ahead=30.0, step=1.0):
    pts = []
    center_pts = []
    cross_walk_pts = []
    
    traveled = 0.0
    while cur_wp is not None and traveled <= dist_ahead:
        t = cur_wp.transform
        pts.append({
            "x": t.location.x,
            "y": t.location.y,
            "z": t.location.z,
            "yaw_deg": t.rotation.yaw,
            "road_id": cur_wp.road_id,
            "lane_id": cur_wp.lane_id,
            "lane_type": str(cur_wp.lane_type),
            "lane_change": str(cur_wp.lane_change),
            "is_junction": bool(cur_wp.is_junction)
        })
        center_pts.append(cur_wp)
        nxts = cur_wp.next(step)
        if len(nxts) == 0:
            break
        for nxt in nxts:
            if nxt.lane_type & carla.LaneType.Crosswalk:
                cross_walk_pts = get_crosswalk_info()
        # 그냥 첫 번째 후보만 따라간다 (fork/분기 무시)
        cur_wp = nxts[0]
        traveled += step
    left_boundaries, right_boundaries, centerlines = left_right_boundaries_from_centerline(center_pts)
    return pts, left_boundaries, right_boundaries, centerlines

def get_vehicle_bbox(ego_vehicle, vehicles, radius=50.0):
    ego_loc = ego_vehicle.get_transform().location

    bboxes = []
    for v in vehicles:
        if v.id == ego_vehicle.id:
            continue  # skip ego

        loc = v.get_transform().location
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dz = loc.z - ego_loc.z
        dist2 = dx*dx + dy*dy + dz*dz
        if dist2 > radius*radius:
            continue

        bb = v.bounding_box           # local bbox
        tf = v.get_transform()        # world pose

        # 우리는 두 가지 표현을 만들 수 있어:
        # (A) center+extent+rotation (compact)
        # (B) 8 corner points in world frame (detailed for visualization / collision check)

        # (A):
        if "vehicle" in v.type_id:
            label = 0
        elif "pedestrian" in v.type_id:
            label = 1
        else:
            label = 2
        bbox_dict = {
            "actor_id": v.id,
            "class": v.type_id,  # e.g. 'vehicle.tesla.model3'
            "world_transform": {
                "location": (tf.location.x, tf.location.y, tf.location.z),
                "rotation": (tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw),
            },
            "extent": (bb.extent.x, bb.extent.y, bb.extent.z),
            "gt_array": np.array([tf.location.x, tf.location.y, tf.location.z, bb.extent.x, bb.extent.y, bb.extent.z, tf.rotation.yaw]),
            "label": label
        }

        # (B) 8 corners 변환까지 계산할 수도 있어.
        # corners in local bbox frame:
        corners_local = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners_local.append(carla.Location(
                        x=bb.location.x + sx*bb.extent.x,
                        y=bb.location.y + sy*bb.extent.y,
                        z=bb.location.z + sz*bb.extent.z
                    ))
        # local -> world transform
        def apply_transform(tf, point):
            # rotate then translate
            yaw = math.radians(tf.rotation.yaw)
            pitch = math.radians(tf.rotation.pitch)
            roll = math.radians(tf.rotation.roll)

            # rotation order in CARLA is roll->pitch->yaw
            # build rotation manually:
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)

            # R = R_yaw * R_pitch * R_roll
            R = [
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr]
            ]

            px = point.x
            py = point.y
            pz = point.z
            wx = R[0][0]*px + R[0][1]*py + R[0][2]*pz + tf.location.x
            wy = R[1][0]*px + R[1][1]*py + R[1][2]*pz + tf.location.y
            wz = R[2][0]*px + R[2][1]*py + R[2][2]*pz + tf.location.z
            return (wx, wy, wz)

        corners_world = [apply_transform(tf, p) for p in corners_local]
        bbox_dict["corners_world"] = corners_world

        bboxes.append(bbox_dict)

    return bboxes

def get_local_lane_sample(world_map, ego_vehicle, vehicles, dist_ahead=30.0, step=1.0):
    """
    ego 차량 위치 근처 lane centerline polyline을 수집해서 리턴.
    - world_map.get_waypoint()로 ego차량 위치의 waypoint를 잡고
    - 그 waypoint에서 차선을 따라 dist_ahead까지 step 간격으로 앞으로 따라가며 좌표 샘플
    """
    ego_loc = ego_vehicle.get_transform().location
    wp_start = world_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    wp_queue = []
    lane_list = []
    bboxes = get_vehicle_bbox(ego_vehicle, vehicles, radius=50.0)
    wp_queue.append(wp_start)
    traveled_lane_ids = set()
    while(len(wp_queue) > 0):
        cur_wp = wp_queue.pop(0)
        lane_key = cur_wp.lane_id
        traveled_lane_ids.add(lane_key)
        if cur_wp.lane_type & carla.LaneType.Sidewalk:
            continue #ignore if sidewalk
        lane_list.append(get_lane_info(world_map, cur_wp, dist_ahead, step))
        if cur_wp.get_left_lane() is not None and cur_wp.get_left_lane().lane_id not in traveled_lane_ids:
            wp_queue.append(cur_wp.get_left_lane())
        if cur_wp.get_right_lane() is not None and cur_wp.get_right_lane().lane_id not in traveled_lane_ids:
            wp_queue.append(cur_wp.get_right_lane())
     
    return lane_list, bboxes

def get_gemap_vis(center_pts, divider_pts, bound_pts, cross_pts, bboxes, ego_vehicle, camera_units, scale = 10.0):
    """
    lane_list: get_local_lane_sample()의 첫 번째 반환값
    ego_vehicle: ego 차량 actor (위치 기준)
    """
    image_size = (1200, 1200)
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    h, w = image_size
    origin = (w // 2, h // 2)  # ego 차량을 중앙에 둠
     
    
    ego_loc = ego_vehicle.get_transform().location
    ego_rot = ego_vehicle.get_transform().rotation
    ego_x, ego_y = ego_loc.x, ego_loc.y
    ego_yaw_deg = ego_rot.yaw

    ego_yaw = math.radians(ego_yaw_deg)

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    

    def world_to_image(x, y):
        """CARLA world 좌표 -> OpenCV 이미지 좌표"""
        dx = x - ego_x
        dy = y - ego_y

        # 2) ego heading을 위로 정렬하기 위한 회전 (-yaw)
        X_body =  cos_yaw * dx + sin_yaw * dy      # 좌(+)/우(-) 축 비슷한 느낌
        Y_body = -sin_yaw * dx + cos_yaw * dy      # 앞(+)/뒤(-) 축

        # 3) 픽셀 좌표로 변환 (OpenCV)
        ix = int(origin[0] + X_body * scale)
        iy = int(origin[1] + Y_body * scale)  # 전방(+)일수록 화면 위쪽(iy 작아짐)

        return ix, iy

    # 차선별 색상 설정
    color_center = (255, 255, 0)  # 노랑
    color_boundaries = (0, 255, 0)      # 초록
    bbox_color = (0,0,255)
    color_crosswalk = (255,0,255)
    

    
    # centerline
    for (x, y, z) in center_pts:
        ix, iy = world_to_image(x, y)
        cv2.circle(img, (ix, iy), 1, color_center, -1)

    # left boundary
    for (x, y, z) in bound_pts:
        ix, iy = world_to_image(x, y)
        cv2.circle(img, (ix, iy), 1, color_boundaries, -1)

    for (x, y, z) in divider_pts:
        ix, iy = world_to_image(x, y)
        cv2.circle(img, (ix, iy), 1, color_boundaries, -1)
    # right boundary
    for (x, y, z) in cross_pts:
        ix, iy = world_to_image(x, y)
        cv2.circle(img, (ix, iy), 1, color_crosswalk, -1)

    def get_top_face_points(corners8):
        # corners8: (8,3) np.array or list
        corners8 = np.array(corners8)  # (8,3)
        # sort by z descending, take top 4
        idxs = np.argsort(corners8[:,2])[::-1]  # 큰 z 먼저
        top4 = corners8[idxs[:4], :]            # (4,3)

        # 이제 이 4점을 (x,y)평면에서 시계방향 순서로 정렬해서 안정화
        # 1) 중심 구하고
        cx = np.mean(top4[:,0])
        cy = np.mean(top4[:,1])
        # 2) 각 점의 각도
        angles = np.arctan2(top4[:,1] - cy, top4[:,0] - cx)
        order = np.argsort(angles)  # -pi..pi 오름차순 (CCW)
        top4_sorted = top4[order]
        return top4_sorted  # shape (4,3), CCW 정렬

    # 변의 중점 계산
    def edge_midpoints(pts4):
        # pts4: (4,3) CCW 정렬된 상부 face
        mids = []
        for i in range(4):
            p0 = pts4[i]
            p1 = pts4[(i+1)%4]
            mid = 0.5*(p0 + p1)
            mids.append(mid)
        return np.array(mids)  # (4,3)
    if bboxes is not None:
        # bboxes is expected to be a tuple/list: (vehicle_list, pedestrian_list)
        ve_list = bboxes[0] if len(bboxes) > 0 and bboxes[0] is not None else []
        pe_list = bboxes[1] if len(bboxes) > 1 and bboxes[1] is not None else []

        # If nothing to draw, skip early
        if len(ve_list) + len(pe_list) > 0:

            # Determine K (length of gt_array per object), assuming at least one exists
            sample_gt = ve_list[0]['gt_array'] if ve_list else pe_list[0]['gt_array']
            K = int(np.asarray(sample_gt).shape[-1])

            def build_gt(list_):
                if not list_:
                    # 0 rows, K columns
                    return np.empty((0, K), dtype=np.float32)
                arr = np.asarray([bb['gt_array'] for bb in list_], dtype=np.float32)
                return arr.reshape(-1, K)

            def build_corners(list_):
                if not list_:
                    # 0 boxes, 8 corners, 3 coords
                    return np.empty((0, 8, 3), dtype=np.float32)
                arr = np.asarray([bb['corners_world'] for bb in list_], dtype=np.float32)
                return arr.reshape(-1, 8, 3)

            ve_bboxes         = build_gt(ve_list)
            pe_bboxes         = build_gt(pe_list)
            ve_bboxes_corners = build_corners(ve_list)
            pe_bboxes_corners = build_corners(pe_list)
            ve_labels         = [bb.get('label', 0) for bb in ve_list]
            pe_labels         = [bb.get('label', 0) for bb in pe_list]

            # Safe vertical stacks (still work if one side is empty)
            if ve_bboxes.size or pe_bboxes.size:
                bboxes_3d = np.vstack([ve_bboxes, pe_bboxes])
            else:
                bboxes_3d = np.empty((0, K), dtype=np.float32)

            if ve_bboxes_corners.size or pe_bboxes_corners.size:
                bboxes_3d_corners = np.vstack([ve_bboxes_corners, pe_bboxes_corners])
            else:
                bboxes_3d_corners = np.empty((0, 8, 3), dtype=np.float32)

            labels = ve_labels + pe_labels
            number = bboxes_3d.shape[0]

            if number > 0:
                # Frustum filter with one or more cameras
                mask = np.zeros((number,), dtype=bool)

                # always use the first camera
                mask = mask | filter_points(
                    bboxes_3d[:, :3],
                    ego_vehicle,
                    camera_units[0]['sensor'],
                    max_distance_m=61,
                    min_distance_m=0.1
                )[0]

                # only use the 2nd camera if it exists
                if len(camera_units) > 1 and camera_units[1].get('sensor') is not None:
                    mask = mask | filter_points(
                        bboxes_3d[:, :3],
                        ego_vehicle,
                        camera_units[1]['sensor'],
                        max_distance_m=61,
                        min_distance_m=0.1
                    )[0]

                # Draw only masked boxes
                for idx, corners8 in enumerate(bboxes_3d_corners):
                    if not mask[idx]:
                        continue

                    # corners8: (8,3)
                    if corners8.shape[0] < 4:
                        continue

                    top4 = get_top_face_points(corners8)  # (4,3)
                    mids4 = edge_midpoints(top4)          # (4,3)

                    # top face corners
                    for (x, y, z) in top4:
                        ix, iy = world_to_image(x, y)
                        cv2.circle(img, (ix, iy), 3, bbox_color, -1)

                    # edge midpoints
                    for (x, y, z) in mids4:
                        ix, iy = world_to_image(x, y)
                        cv2.circle(img, (ix, iy), 2, bbox_color, -1)

                    # outline
                    for i in range(4):
                        x0, y0, _ = top4[i]
                        x1, y1, _ = top4[(i + 1) % 4]
                        ix0, iy0 = world_to_image(x0, y0)
                        ix1, iy1 = world_to_image(x1, y1)
                        cv2.line(img, (ix0, iy0), (ix1, iy1), bbox_color, 1)
        # ve_bboxes = np.array([bb['gt_array'] for bb in bboxes[0]])
        # ve_bboxes_corners = np.array([bb['corners_world'] for bb in bboxes[0]])
        # ve_bboxes_label = np.array([bb['label'] for bb in bboxes[0]])
        # pe_bboxes = np.array([bb['gt_array'] for bb in bboxes[1]])
        # pe_bboxes_corners = np.array([bb['corners_world'] for bb in bboxes[1]])
        # pe_bboxes_label = np.array([bb['label'] for bb in bboxes[1]])
        # bboxes_3d = np.concatenate([ve_bboxes, pe_bboxes], axis=0)
        # bboxes_3d_corners = np.concatenate([ve_bboxes_corners, pe_bboxes_corners], axis=0)
        # number = bboxes_3d.shape[0]
        # mask = np.zeros((number,), dtype=bool)
        # mask = mask | filter_points(bboxes_3d[:, :3], ego_vehicle, camera_units[0]['sensor'], max_distance_m=61, min_distance_m=0.1)[0]
        # if len(camera_units) >1:
        #     mask = mask | filter_points(bboxes_3d[:, :3], ego_vehicle, camera_units[1]['sensor'], max_distance_m=61, min_distance_m=0.1)[0]
        # for idx, bbox in enumerate(bboxes_3d_corners):
        # # bbox: (8,3) world corners
        #     if mask[idx] == True:
        #         if bbox.shape[0] < 4:
        #             continue  # 안전장치

        #         top4 = get_top_face_points(bbox)           # (4,3)
        #         mids4 = edge_midpoints(top4)               # (4,3)

        #         # 코너 점 표시
        #         for (x,y,z) in top4:
        #             ix, iy = world_to_image(x, y)
        #             cv2.circle(img, (ix, iy), 3, bbox_color, -1)

        #         # 변 중점 표시
        #         for (x,y,z) in mids4:
        #             ix, iy = world_to_image(x, y)
        #             cv2.circle(img, (ix, iy), 2, bbox_color, -1)

        #         # 원하면 윤곽선도 그릴 수 있음 (top face 폴리라인)
        #         for i in range(4):
        #             x0,y0,_ = top4[i]
        #             x1,y1,_ = top4[(i+1)%4]
        #             ix0, iy0 = world_to_image(x0, y0)
        #             ix1, iy1 = world_to_image(x1, y1)
        #             cv2.line(img, (ix0, iy0), (ix1, iy1), bbox_color, 1)

    # 중심에 ego 차량 위치 마커 표시
    cv2.circle(img, origin, 4, (0, 0, 255), -1)
    cv2.putText(img, "EGO", (origin[0]+5, origin[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    return img

def filter_by_cameras(camera, ego_vehicle, center_pts, divider_pts, bound_pts, cross_pts, masks, max_dist=61, min_dist= 0.1):

    center_mask,_ = filter_points(center_pts, ego_vehicle=ego_vehicle, sensor_actor=camera, max_distance_m=61, min_distance_m=0.1)
    divider_mask,_ = filter_points(divider_pts, ego_vehicle=ego_vehicle, sensor_actor=camera, max_distance_m=61, min_distance_m=0.1)
    bound_mask,_ = filter_points(bound_pts, ego_vehicle=ego_vehicle, sensor_actor=camera, max_distance_m=61, min_distance_m=0.1)
    cross_mask,_ = filter_points(cross_pts, ego_vehicle=ego_vehicle, sensor_actor=camera, max_distance_m=61, min_distance_m=0.1)
    return (masks[0].astype(bool)| center_mask), (masks[1].astype(bool)| divider_mask), (masks[2].astype(bool) | bound_mask), (masks[3].astype(bool) | cross_mask)

def transform_points_world_to_sensor(points_world, sensor_transform):
    """
    points_world: (N,3) np.ndarray in world frame [x,y,z]
    sensor_transform: carla.Transform of the sensor (location + rotation in world)
    return: (N,3) np.ndarray of the same points expressed in the *sensor* coordinate frame
    """
    # sensor world pose
    sensor_loc = sensor_transform.location
    sensor_rot = sensor_transform.rotation  # pitch(y), yaw(z), roll(x) in degrees (CARLA convention)
    
    # Build rotation matrix world->sensor
    # CARLA rotation order is yaw(Z), pitch(Y), roll(X), all left-handed UE4 style.
    # We'll convert world point to sensor by:
    #   p_rel = p_world - sensor_loc
    #   p_sensor = R_world_to_sensor * p_rel
    #
    # where R_world_to_sensor = R_sensor_world.T  (transpose / inverse of sensor->world)

    # Step 1: build sensor->world rotation matrix
    cy = math.cos(math.radians(sensor_rot.yaw))
    sy = math.sin(math.radians(sensor_rot.yaw))
    cp = math.cos(math.radians(sensor_rot.pitch))
    sp = math.sin(math.radians(sensor_rot.pitch))
    cr = math.cos(math.radians(sensor_rot.roll))
    sr = math.sin(math.radians(sensor_rot.roll))

    # UE4 / CARLA convention for forward(+X), right(+Y), up(+Z):
    # Rotation matrix from sensor->world:
    # reference: CARLA docs / common community snippet
    R_sw = np.array([
        [cp*cy,            cp*sy,            sp     ],
        [cy*sp*sr - cr*sy, sy*sp*sr - cr*cy, -cp*sr ],
        [-cr*cy*sp - sr*sy, -cr*sy*sp - sr*cy, cp*cr]
    ], dtype=np.float32)

    # world->sensor is transpose
    R_ws = R_sw.T

    # translation world->sensor
    t = np.array([sensor_loc.x, sensor_loc.y, sensor_loc.z], dtype=np.float32)

    # apply transform
    p_rel = points_world - t[None, :]          # (N,3)
    p_sensor = p_rel @ R_ws.T                  # (N,3)
    return p_sensor


def filter_points(points_world,
                  ego_vehicle,
                  sensor_actor,
                  max_distance_m=50.0,
                  min_distance_m=0.5):
    """
    points_world: (N,3) np.ndarray of 3D points in world coordinates.
    ego_vehicle: carla.Actor of the ego vehicle.
    sensor_actor: carla.Actor (e.g. camera sensor) mounted on the ego.
    max_distance_m: keep only points within this distance from ego_vehicle.
    horiz_fov_deg, vert_fov_deg: sensor field of view in degrees.
    min_distance_m: optional near clip distance in meters (avoid points basically on top of sensor)

    returns:
        mask (N,) boolean, and filtered_points_world (M,3)
    """

    # 1) distance filter w.r.t ego vehicle center
    ego_loc = ego_vehicle.get_transform().location
    ego_xyz = np.array([ego_loc.x, ego_loc.y, ego_loc.z], dtype=np.float32)

    # compute Euclidean distance
    diff = points_world - ego_xyz[None, :]
    dist = np.linalg.norm(diff, axis=1)

    dist_mask = (dist <= max_distance_m) & (dist >= min_distance_m)

    # early prune
    pts_after_dist = points_world[dist_mask]

    if pts_after_dist.shape[0] == 0:
        # nothing survives
        return np.zeros(points_world.shape[0], dtype=bool), pts_after_dist

    # 2) FOV filter using sensor pose
    sensor_tf = sensor_actor.get_transform()
    pts_sensor = transform_points_world_to_sensor(pts_after_dist, sensor_tf)

    Xc = pts_sensor[:, 0]
    Yc = pts_sensor[:, 1]
    Zc = pts_sensor[:, 2]

    # point must be in front of sensor
    front_mask = Xc > 0.0

    # horizontal angle
    theta_h = np.arctan2(Yc, Xc)  # radians
    # vertical angle (note: CARLA cam has +Z down, but we're just symmetric in angle)
    theta_v = np.arctan2(Zc, Xc)  # radians
    
    attrs = sensor_actor.attributes
    if "fov" in attrs:
        hfov_deg = float(attrs["fov"])
    else:
        hfov_deg = 90.0  # 기본값 fallback

    # image 크기 정보로 vertical fov 추정
    if "image_size_x" in attrs and "image_size_y" in attrs:
        image_w = int(attrs["image_size_x"])
        image_h = int(attrs["image_size_y"])
        aspect = image_h / image_w
        vfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * aspect))
    else:
        vfov_deg = hfov_deg * 0.75  # 대략적 fallback

    h_half = math.radians(hfov_deg) / 2.0
    v_half = math.radians(vfov_deg) / 2.0

    fov_mask_local = (
        front_mask &
        (np.abs(theta_h) <= h_half) &
        (np.abs(theta_v) <= v_half)
    )

    # map fov_mask_local back to original indexing
    final_mask = np.zeros(points_world.shape[0], dtype=bool)
    surviving_idx = np.nonzero(dist_mask)[0]          # indices in original array that passed dist
    final_mask[surviving_idx[fov_mask_local]] = True  # keep only also in FOV

    return final_mask, points_world[final_mask]

def get_camera_intrinsic(sensor):
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))
    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    return calibration

def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix  

def get_camera_extrinsic(sensor, egovehicle_snapshot):
    # sensor to ego vehicle
    sensor_tf = sensor.get_transform() #w2s 
    ego_tf = egovehicle_snapshot.get_transform()
    mat_se = np.linalg.inv(get_matrix(sensor_tf))@get_matrix(ego_tf)  #s2w * w2e = s2e
    

    # ego vehicle to world
    # sensor to world
    return mat_se


def put_loc_info(dic, transform):
    mat = get_matrix(transform)
    dic['e2g_translation'] = np.array(mat[:3, 3]).reshape(3)
    dic['e2g_rotation'] = np.array(mat[:3, :3])
    return dic

def put_camera_info(dic, camera, egovehicle_snapshot, frame_name):
    cam_tf = camera.get_transform()
    dic = put_loc_info(dic, cam_tf)
    dic['intrinsics'] = np.array(get_camera_intrinsic(camera))
    dic['extrinsics'] = np.array(get_camera_extrinsic(camera, egovehicle_snapshot))
    #sdic['camera_fov'] = float(camera.attributes.get('fov', 90.0))
    dic['img_fpath'] = frame_name
    return dic  



def get_gemap_gt(camera_units, center_pts_gt, divider_pts_gt, bound_pts_gt, cross_pts_gt, bboxes, egovehicle_snapshot, frame_name, count, timestamp, log_id, token):
    gt = dict()

    gt = put_loc_info(gt, egovehicle_snapshot.get_transform())
    gt['cams'] = dict()
    gt['cams'][camera_units[0]['name']] = put_camera_info(dict(), camera_units[0]['sensor'], egovehicle_snapshot, os.path.join(camera_units[0]['dirs']['data_dir'], frame_name+'.png'))
    gt['cams'][camera_units[1]['name']] = put_camera_info(dict(), camera_units[1]['sensor'], egovehicle_snapshot, os.path.join(camera_units[1]['dirs']['data_dir'], frame_name+'.png'))
    gt['lidar_path'] = os.path.join(camera_units[2]['dirs']['data_dir'], frame_name+'.bin')
    gt['annotation'] = {
        'divider': divider_pts_gt,
        'ped_crossing': cross_pts_gt,
        'centerline': center_pts_gt,
        'boundary': bound_pts_gt,
    }
    gt['token'] = token
    gt['log_id'] = log_id
    gt['sample_idx'] = count
    gt['timestamp'] = timestamp
    import pdb; pdb.set_trace() 
    ve_bboxes = np.array([bb['gt_array'] for bb in bboxes[0]])
    ve_bboxes_label = np.array([bb['label'] for bb in bboxes[0]])
    pe_bboxes = np.array([bb['gt_array'] for bb in bboxes[1]])
    pe_bboxes_label = np.array([bb['label'] for bb in bboxes[1]])
    
    gt['instances'] = {
        '3d_bboxes': np.stack([ve_bboxes, pe_bboxes], axis=0),
        '3d_labels': np.stack([ve_bboxes_label, pe_bboxes_label], axis=0)
    }
    number = gt['instances']['3d_bboxes'].shape[0]
    mask = np.zeros((number,), dtype=bool)
    mask = mask | filter_points(gt['instances']['3d_bboxes'][:, :3], egovehicle_snapshot, camera_units[0]['sensor'], max_distance_m=61, min_distance_m=0.1)[0]
    mask = mask | filter_points(gt['instances']['3d_bboxes'][:, :3], egovehicle_snapshot, camera_units[1]['sensor'], max_distance_m=61, min_distance_m=0.1)[0]
    gt['instances']['3d_bboxes'] = gt['instances']['3d_bboxes'][mask]
    gt['instances']['3d_labels'] = gt['instances']['3d_labels'][mask]
    return gt

    

    
