import argparse
import os
import time
import weakref
import math

import numpy as np
import cv2
import carla

# your utils (reuse the same ones you already import in manual_control.py)
# from team_code_autopilot.utils import carla_vehicle_annotator as cva
# from team_code_autopilot.utils import gemap_annotator as ga
# from team_code_autopilot.utils import test_xodr as tx
import carla_vehicle_annotator as cva
import gemap_annotator as ga
import test_xodr as tx


# ---------- Camera setups (same as data_label_generate_ours.py) ----------
CAMERA_SETUPS = {
    'RGB_1': {
        'enabled': True,
        'location': (0.5,  0.10, 2.2),
        'rotation': (-8.0, 0.0, 0.0),
    },
    'RGB_2': {
        'enabled': True,
        'location': (0.5, -0.10, 2.2),
        'rotation': (-8.0, 0.0, 0.0),
    },
    'lidar_1': {
        'enabled': True,
        'location': (0.5, 0.0, 2.2),
        'rotation': (0.0, 0.0, 0.0),
    }
}



def _prepare_hdmap_points(world):
    xodr = world.get_map().to_opendrive()
    center_pts, divider_pts, bound_pts, cross_pts, idxes = tx.extract_waypoints(xodr)
    return center_pts, divider_pts, bound_pts, cross_pts, idxes


def _find_ego_by_role(world, role_name):
    actors = world.get_actors().filter("vehicle.*")
    for a in actors:
        if a.attributes.get("role_name", "") == role_name:
            return a
    return None


def _spawn_rgb(world, parent, image_w, image_h, loc=(0.5, 0.0, 2.2), rot=(-8.0, 0.0, 0.0)):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_w))
    bp.set_attribute("image_size_y", str(image_h))
    bp.set_attribute("sensor_tick", "0.0")  # run every tick

    t = carla.Transform(
        carla.Location(x=loc[0], y=loc[1], z=loc[2]),
        carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
    )
    sensor = world.try_spawn_actor(bp, t, attach_to=parent)
    return sensor

def simple_hdmap_image(center_pts, divider_pts, bound_pts, cross_pts,
                       ego_snap,
                       img_size=(800, 800),
                       pixels_per_meter=4.0):
    """
    마스크된 HD map 포인트들을 'ego 차량 좌표계'로 변환해서 그리는 간단 vis.

    - world (x, y) -> ego frame (X_body, Y_body)
      * ego는 이미지 중앙에 위치
      * Y_body: 앞(+), X_body: 왼쪽(+)
    - min/max 기반 자동 스케일링 없음
    - 고정 pixels_per_meter 로만 스케일
    """

    h, w = img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # ego pose 가져오기 (cva.snap_processing 의 snapshot 객체라고 가정)
    ego_tf = ego_snap.get_transform()
    ego_loc = ego_tf.location
    ego_rot = ego_tf.rotation

    ego_x = ego_loc.x
    ego_y = ego_loc.y
    ego_yaw = math.radians(ego_rot.yaw)

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    cx = w // 2   # 이미지 중앙 (ego 위치)
    cy = h // 2

    def world_to_ego_pixel(x, y):
        # 1) world → ego 평면
        dx = x - ego_x
        dy = y - ego_y

        # X_body =  cos_yaw * dx + sin_yaw * dy   # 좌(+)/우(-)
        # Y_body = -sin_yaw * dx + cos_yaw * dy   # 앞(+)/뒤(-)
        Y_body =  cos_yaw * dx + sin_yaw * dy       # forward(+)
        X_body = -sin_yaw * dx + cos_yaw * dy       # left(+)


        # 2) ego 좌표 → pixel (고정 스케일)
        ix = int(round(cx + X_body * pixels_per_meter))
        iy = int(round(cy - Y_body * pixels_per_meter))  # 앞(+Y_body)가 위쪽으로 가도록 - 부호

        return ix, iy

    def draw_points(pts, color, radius=1):
        if pts is None or len(pts) == 0:
            return
        pts = np.asarray(pts)
        xs = pts[:, 0]
        ys = pts[:, 1]
        for x, y in zip(xs, ys):
            ix, iy = world_to_ego_pixel(x, y)
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(img, (ix, iy), radius, color, -1)

    color_center   = (0, 255, 255)  # 중앙선
    color_divider  = (0, 255, 0)    # 차선 분리선
    color_bound    = (255, 255, 0)  # 차선 경계
    color_cross    = (255, 0, 255)  # 횡단보도

    draw_points(center_pts,  color_center)
    draw_points(divider_pts, color_divider)
    draw_points(bound_pts,   color_bound)
    draw_points(cross_pts,   color_cross)

    # ego 위치 표시 (중앙)
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.putText(img, "EGO", (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return img


def main():
    ap = argparse.ArgumentParser("Standalone HD-Map Visualizer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--role", default="hero", help="Ego role_name to attach to")
    ap.add_argument("--res", default="1936x1216", help="RGB sensor resolution WxH")
    ap.add_argument("--dist", type=float, default=61.0, help="max distance for masks/bboxes")
    ap.add_argument("--save-dir", default=None, help="optional directory to save frames")
    args = ap.parse_args()

    w, h = [int(x) for x in args.res.lower().split("x")]
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    # Parse the map once
    center_pts, divider_pts, bound_pts, cross_pts, idxes = _prepare_hdmap_points(world)

    cam = None
    ego = _find_ego_by_role(world, args.role)
    if ego is None:
        print(f"[hdmap_vis] Waiting for ego with role_name='{args.role}' ...")
        # simple wait loop until ego appears
        while ego is None:
            world.wait_for_tick(1.0)
            ego = _find_ego_by_role(world, args.role)

    camera_units = []
    sensors = []  # for cleanup

    # RGB sensors (we only need RGB_1, RGB_2 for gemap_vis; lidar_1는 생략 가능)
    for cam_name, cam_cfg in CAMERA_SETUPS.items():
        if not cam_cfg.get('enabled', True):
            continue
        if not cam_name.startswith('RGB'):
            continue  # lidar는 HD map vis에는 필요 없음 (원하면 따로 추가 가능)

        loc = cam_cfg.get('location', (0.5, 0.0, 2.2))
        rot = cam_cfg.get('rotation', (-8.0, 0.0, 0.0))

        sensor = _spawn_rgb(world, ego, w, h, loc=loc, rot=rot)
        if sensor is None:
            print(f"[hdmap_vis] Failed to spawn {cam_name}")
            continue

        sensors.append(sensor)
        camera_units.append({
            'name': cam_name,
            'sensor': sensor,
            'depth_sensor': None,  # datagen에서는 depth도 있지만 여기서는 불필요
            'dirs': {}
        })

    if not camera_units:
        print("[hdmap_vis] No RGB cameras could be spawned; exiting.")
        return

    print("[hdmap_vis] Running. Press Ctrl+C to quit.")
    frame_idx = 0

    try:
        while True:
            snapshot = world.wait_for_tick(2.0)
            if snapshot is None:
                continue

            actors = world.get_actors()
            if ego is None or not ego.is_alive:
                ego = _find_ego_by_role(world, args.role)
                if ego is None:
                    continue

            # Ego snapshot
            try:
                ego_snap = cva.snap_processing([ego], snapshot)[0]
            except Exception:
                continue

            vehicles = [a for a in actors.filter("vehicle.*") if a.id != ego.id]
            walkers  = list(actors.filter("walker.*"))
            vehicle_snaps = cva.snap_processing(vehicles, snapshot) if vehicles else []
            walker_snaps  = cva.snap_processing(walkers,  snapshot) if walkers  else []

            # ----- HD map masks (center/divider/bound/cross) -----
            center_mask  = np.zeros(len(center_pts),  dtype=bool)
            divider_mask = np.zeros(len(divider_pts), dtype=bool)
            bound_mask   = np.zeros(len(bound_pts),   dtype=bool)
            cross_mask   = np.zeros(len(cross_pts),   dtype=bool)

            # datagen처럼 RGB_1, RGB_2 각각으로 필터링
            for unit in camera_units:
                if not unit['name'].startswith('RGB'):
                    continue
                cam_actor = unit['sensor']
                if cam_actor is None or not cam_actor.is_alive:
                    continue
                center_mask, divider_mask, bound_mask, cross_mask = ga.filter_by_cameras(
                    camera=cam_actor,
                    ego_vehicle=ego_snap,
                    center_pts=center_pts,
                    divider_pts=divider_pts,
                    bound_pts=bound_pts,
                    cross_pts=cross_pts,
                    masks=(center_mask, divider_mask, bound_mask, cross_mask),
                    max_dist=args.dist,
                    min_dist=0.1
                )

            # ----- 3D bboxes -----
            veh_bboxes3d  = ga.get_vehicle_bbox(ego_snap, vehicle_snaps, radius=args.dist)
            walk_bboxes3d = ga.get_vehicle_bbox(ego_snap, walker_snaps,  radius=args.dist)

            # # ----- GEMAP visualization (exactly like datagen) -----
            # vis = ga.get_gemap_vis(
            #     center_pts[center_mask],
            #     divider_pts[divider_mask],
            #     bound_pts[bound_mask],
            #     cross_pts[cross_mask],
            #     (veh_bboxes3d, walk_bboxes3d),
            #     ego_snap,
            #     camera_units  # 여기 중요!
            # )

            # vis = simple_hdmap_image(
            #     center_pts[center_mask],
            #     divider_pts[divider_mask],
            #     bound_pts[bound_mask],
            #     cross_pts[cross_mask],
            #     img_size=(800, 800)   # 원하면 바꿔도 됨
            # )

            vis = simple_hdmap_image(
                center_pts[center_mask],
                divider_pts[divider_mask],
                bound_pts[bound_mask],
                cross_pts[cross_mask],
                ego_snap=ego_snap,
                img_size=(800, 800),
                pixels_per_meter=4.0   # 한 칸당 몇 픽셀인지 조절 가능
            )

            cv2.imshow("HDMap_GT (standalone)", vis)
            cv2.waitKey(1)

            if args.save_dir:
                out_path = os.path.join(args.save_dir, f"hdmap_{frame_idx:08d}.png")
                cv2.imwrite(out_path, vis)
                frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cam_ref() is not None:
                cam_ref().destroy()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[hdmap_vis] Bye.")


if __name__ == "__main__":
    main()