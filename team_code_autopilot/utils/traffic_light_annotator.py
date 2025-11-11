# -*- coding: utf-8 -*-
"""
Traffic Light Data Labeling Script for CARLA
Generates 2D bounding box labels for traffic lights with state classification
Based on data_label_generate_part2.py structure
"""

import os
import random
import queue
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pygame
from scipy.spatial import cKDTree

import carla

# ---------- Client / World ----------
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
original_settings = world.get_settings()

# ---------- Parameters ----------
num_vehicles = 30  # Reduced vehicles for better traffic light visibility
tm_port = 8000
fps = 10

# Data collection efficiency settings
FRAME_SKIP = 3  # Save only every Nth frame (1=save all, 3=save 1 out of 3)
MAX_FRAMES_PER_CAMERA = 800  # Maximum frames to save per camera per scene
SKIP_DUPLICATE_STATES = True  # Skip if traffic light states haven't changed

# Display settings
DISPLAY_ENABLED = True  # Set to False to disable pygame window
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Traffic Light Classes (YOLO format)
TRAFFIC_LIGHT_CLASSES = {
    'Red': 0,
    'Yellow': 1,
    'Green': 2,
    'Unknown': 3,
}

VISUALIZATION_COLORS = {
    'Red': (255, 0, 0),
    'Yellow': (255, 255, 0),
    'Green': (0, 255, 0),
    'Unknown': (128, 128, 128),
}

# --- Traffic Light filtering parameters ---
MAX_DIST_TRAFFIC_LIGHT = 100.0  # meters
MIN_BBOX_AREA = 5 * 5           # px^2, minimum area (더 작은 신호등도 허용)
MIN_BBOX_WIDTH = 3              # minimum width in pixels (완화)
MIN_BBOX_HEIGHT = 5             # minimum height in pixels (완화)

# Traffic Light Head Center (신호등 전등 부분 위치) - fallback용
TL_HEAD_CENTER_Z = 2.5          # 기둥 하단으로부터 헤드 중심까지 높이 (3.5~4.5 조정 가능)
TL_EXTENT_X = 0.25              # half-size: 가로 0.5m
TL_EXTENT_Y = 0.25              # half-size: 깊이 0.5m
TL_EXTENT_Z = 0.60              # half-size: 세로 1.2m

# Traffic Light Level Bounding Boxes (맵마다 자동으로 실제 크기 반영)
TL_LEVEL_BBS = []
TL_LEVEL_CENTERS = None
TL_LEVEL_KD = None
TLID_TO_BBIDX = {}

# Camera setups - focusing on front views for traffic lights
CAMERA_SETUPS = {
    'front': {
        'enabled': True,
        'location': (0.5, 0.0, 2.2),
        'rotation': (-8.0, 0.0, 0.0),
    },
    'front_left': {
        'enabled': True,
        'location': (0.5, -0.5, 2.2),
        'rotation': (-8.0, -45.0, 0.0),  # 왼쪽은 -45도
    },
    'front_right': {
        'enabled': True,
        'location': (0.5, 0.5, 2.2),
        'rotation': (-8.0, 45.0, 0.0),   # 오른쪽은 +45도
    },
}

# ---------- Helper Classes ----------
class CarlaSyncMode(object):
    def __init__(self, world, sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
        # first: world tick
        make_queue(self.world.on_tick)
        # then: all sensors
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout=1.0):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(d.frame == self.frame for d in data), "Frame mismatch across sensors"
        return data

    def __exit__(self, *args):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        """Wait until we get data for *this* world frame."""
        while True:
            data = sensor_queue.get(timeout=timeout)
            if hasattr(data, "frame"):
                if data.frame == self.frame:
                    return data


def create_camera_transform(config):
    loc_x, loc_y, loc_z = config.get('location', (0.5, 0.0, 2.2))
    rot_pitch, rot_yaw, rot_roll = config.get('rotation', (-8.0, 0.0, 0.0))
    location = carla.Location(x=loc_x, y=loc_y, z=loc_z)
    rotation = carla.Rotation(pitch=rot_pitch, yaw=rot_yaw, roll=rot_roll)
    return carla.Transform(location, rotation)


def build_projection_matrix(w, h, fov):
    """Build camera projection matrix"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_traffic_light_state(traffic_light):
    """Get traffic light state as string"""
    state = traffic_light.get_state()
    if state == carla.TrafficLightState.Red:
        return 'Red'
    elif state == carla.TrafficLightState.Yellow:
        return 'Yellow'
    elif state == carla.TrafficLightState.Green:
        return 'Green'
    else:
        return 'Unknown'


def precompute_tl_level_bbs(world):
    """
    Precompute traffic light level bounding boxes from CARLA map
    Maps each traffic light actor to its corresponding level BB
    """
    global TL_LEVEL_BBS, TL_LEVEL_CENTERS, TL_LEVEL_KD, TLID_TO_BBIDX
    
    TL_LEVEL_BBS = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    if not TL_LEVEL_BBS:
        print("[LevelBB] No traffic-light level BBs found")
        return
    
    print(f"[LevelBB] Found {len(TL_LEVEL_BBS)} traffic light level bounding boxes")
    
    TL_LEVEL_CENTERS = np.array([
        [bb.location.x, bb.location.y, bb.location.z] 
        for bb in TL_LEVEL_BBS
    ], dtype=np.float32)
    TL_LEVEL_KD = cKDTree(TL_LEVEL_CENTERS)

    TLID_TO_BBIDX = {}
    for tl in world.get_actors().filter('traffic.traffic_light'):
        loc = tl.get_transform().location
        d, idx = TL_LEVEL_KD.query([loc.x, loc.y, loc.z], k=1)
        TLID_TO_BBIDX[tl.id] = int(idx)
        if d > 5.0:  # Warning if matching is far
            print(f"[LevelBB] Warning: TL actor {tl.id} matched to BB at distance {d:.1f}m")


def get_bb_world_corners(bb):
    """
    Get 8 world corners of a CARLA BoundingBox
    """
    ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z
    local = [
        carla.Location(+ex, +ey, +ez), carla.Location(+ex, +ey, -ez),
        carla.Location(+ex, -ey, +ez), carla.Location(+ex, -ey, -ez),
        carla.Location(-ex, +ey, +ez), carla.Location(-ex, +ey, -ez),
        carla.Location(-ex, -ey, +ez), carla.Location(-ex, -ey, -ez),
    ]
    rot = getattr(bb, "rotation", carla.Rotation(0, 0, 0))
    tf = carla.Transform(bb.location, rot)
    return [tf.transform(p) for p in local]


def project_points(camera, points_world, image_w, image_h):
    """
    Project multiple 3D world points to 2D image coordinates
    Returns list of (u, v) tuples
    """
    K = build_projection_matrix(image_w, image_h, float(camera.attributes['fov']))
    w2c = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float32)
    out = []
    
    for pw in points_world:
        p = np.array([pw.x, pw.y, pw.z, 1.0], np.float32)
        pc = w2c @ p  # X forward, Y right, Z up
        X, Y, Z = pc[0], pc[1], pc[2]
        if X <= 0:
            continue
        u = K[0, 0] * (Y / X) + K[0, 2]
        v = K[1, 1] * (-Z / X) + K[1, 2]
        out.append((u, v))
    
    return out


def tl_bbox_from_levelbb(tl_actor, camera, image_w, image_h, 
                         head_only=True,
                         head_height_m=1.0,        # 헤드 전체 높이를 m로 (예: 0.9~1.2)
                         head_top_margin_m=0.05):  # 아주 약간 여유
    """
    Generate 2D bbox from traffic light level BB (actual mesh size from map)
    
    Args:
        tl_actor: traffic light actor
        camera: camera sensor
        image_w, image_h: image dimensions
        head_only: if True, only use top portion of BB (head, not pole)
        head_height_m: absolute height of head region in meters (not ratio!)
        head_top_margin_m: small margin from top in meters
    
    Returns:
        bbox [[min_x, min_y], [max_x, max_y]] or None
    """
    if tl_actor.id not in TLID_TO_BBIDX:
        return None
    
    bb = TL_LEVEL_BBS[TLID_TO_BBIDX[tl_actor.id]]
    ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z
    rot = getattr(bb, "rotation", carla.Rotation(0, 0, 0))
    tf = carla.Transform(bb.location, rot)

    if head_only:
        # 로컬 z축: -ez(아래) ~ +ez(위). '상단에서 head_height_m 만큼' 슬랩
        z_top = ez - min(head_top_margin_m, ez * 0.2)     # 상단 살짝 여유
        z_bottom = max(-ez, z_top - head_height_m)        # 절대 높이로 아래 경계
        
        # fallback: ez가 너무 작아 head_height_m을 못 담으면 최소 0.3m 유지
        if z_bottom >= z_top:
            z_bottom = z_top - min(0.3, 2 * ez * 0.5)

        corners = []
        for sx in (-ex, +ex):
            for sy in (-ey, +ey):
                for sz in (z_bottom, z_top):
                    corners.append(tf.transform(carla.Location(sx, sy, sz)))
    else:
        # 전체 BB 사용
        corners = get_bb_world_corners(bb)

    pts2d = project_points(camera, corners, image_w, image_h)
    if len(pts2d) < 2:
        return None
    
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    min_x = max(0, np.min(xs))
    max_x = min(image_w - 1, np.max(xs))
    min_y = max(0, np.min(ys))
    max_y = min(image_h - 1, np.max(ys))
    
    # (선택) 아래쪽을 살짝 내려서 포함 범위 확장
    pad_down_px = int(0.08 * (max_y - min_y))  # 8% 정도
    max_y = min(image_h - 1, max_y + pad_down_px)
    
    if max_x - min_x < MIN_BBOX_WIDTH or max_y - min_y < MIN_BBOX_HEIGHT:
        return None
    if (max_x - min_x) * (max_y - min_y) < MIN_BBOX_AREA:
        return None
    
    return np.array([[min_x, min_y], [max_x, max_y]], dtype=np.float32)


def project_point_to_image(camera, point_world, image_w, image_h):
    """
    Project a 3D world point to 2D image coordinates
    Returns (u, v, depth) or None if behind camera or out of bounds
    """
    K = build_projection_matrix(image_w, image_h, float(camera.attributes['fov']))
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    p = np.array([point_world.x, point_world.y, point_world.z, 1.0], np.float32)
    pc = world_2_camera @ p  # sensor frame: X forward, Y right, Z up
    X, Y, Z = pc[0], pc[1], pc[2]
    
    if X <= 0:  # behind camera
        return None

    u = K[0, 0] * (Y / X) + K[0, 2]
    v = K[1, 1] * (-Z / X) + K[1, 2]
    
    if u < -100 or u > image_w + 100 or v < -100 or v > image_h + 100:
        return None
    
    return float(u), float(v), float(X)


def crop_around(depth_array, cx, cy, bw, bh, shrink=0.5, min_size=8):
    """
    Crop a small ROI around center point (cx, cy)
    ROI size is shrink * bbox_size, with minimum min_size
    """
    H, W = depth_array.shape
    roi_w = max(min_size, int(bw * shrink))
    roi_h = max(min_size, int(bh * shrink))
    x1 = max(0, int(cx - roi_w * 0.5))
    y1 = max(0, int(cy - roi_h * 0.5))
    x2 = min(W, int(cx + roi_w * 0.5))
    y2 = min(H, int(cy + roi_h * 0.5))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return depth_array[y1:y2, x1:x2]


def extract_depth(depth_image):
    """
    Extract depth values from CARLA depth image
    Returns depth in meters as numpy array
    """
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    h, w = depth_image.height, depth_image.width
    buf = raw.reshape((h, w, 4)).astype(np.float32)  # BGRA 순서
    
    # CARLA 이미지는 BGRA 순서 → R, G, B 분리
    B = buf[:, :, 0]
    G = buf[:, :, 1]
    R = buf[:, :, 2]
    
    # CARLA depth encoding: R + G*256 + B*256*256 (R이 LSB)
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0**3 - 1)
    
    # CARLA depth camera: 0~1000m 매핑
    depth_meters = normalized * 1000.0
    
    return depth_meters


def check_occlusion(bbox, traffic_light, camera, depth_array, margin=5.0, debug=False):
    """
    Robust occlusion check for diagonal cameras:
    - Uses small ROI around projected 'head center' instead of full bbox
    - Avoids false positives from background in diagonal views
    - Uses ratio-based detection: checks if enough pixels are "closer" (blocking)
    
    Args:
        bbox: [[min_x, min_y], [max_x, max_y]]
        traffic_light: CARLA traffic light actor
        camera: CARLA camera sensor
        depth_array: depth image as numpy array (H, W) in meters
        margin: base depth margin in meters
        debug: print debug information
    
    Returns:
        True if visible (not occluded), False if occluded
    """
    # 실제 거리(헤드 중심)
    camera_loc = camera.get_transform().location
    head_world = traffic_light.get_transform().transform(
        carla.Location(0.0, 0.0, TL_HEAD_CENTER_Z)
    )
    actual_distance = camera_loc.distance(head_world)

    # bbox 정리
    x1, y1 = float(bbox[0][0]), float(bbox[0][1])
    x2, y2 = float(bbox[1][0]), float(bbox[1][1])
    H, W = depth_array.shape
    x1i = max(0, min(W - 1, int(x1)))
    x2i = max(0, min(W,     int(x2)))  # 끝 인덱스는 슬라이싱에서 배타적
    y1i = max(0, min(H - 1, int(y1)))
    y2i = max(0, min(H,     int(y2)))
    
    if x2i <= x1i or y2i <= y1i:
        if debug:
            print("  [Occ] invalid bbox")
        return False

    bw = x2i - x1i
    bh = y2i - y1i

    # 헤드 중심을 이미지로 투영 → 그 근처만 확인
    proj = project_point_to_image(camera, head_world, W, H)
    if proj is None:
        if debug:
            print("  [Occ] head center behind/out")
        return False
    u0, v0, _ = proj

    # 헤드 중심 주변의 작은 ROI만 크롭
    roi = crop_around(depth_array, u0, v0, bw, bh, shrink=0.5, min_size=8)
    if roi is None or roi.size == 0:
        if debug:
            print("  [Occ] empty ROI")
        return False

    # 동적 마진
    dyn = max(margin, max(2.0, actual_distance * 0.15))

    d = roi.reshape(-1)
    
    # 가림판(더 가까운 물체)가 얼만큼 차지하는지 비율로 판단
    # (actual_distance - dyn)보다 작은 픽셀 비율
    closer_ratio = np.mean(d < (actual_distance - dyn))
    
    # 신호등(또는 그 근처)이 실제거리 근처에 존재하는지
    near_ratio = np.mean((d > (actual_distance - dyn)) & (d < (actual_distance + dyn)))

    # 결정 규칙:
    # 1) 가까운 물체가 ROI의 25% 이상을 차지하면 → 가림(occluded)
    # 2) 아니고, 실제 거리 근처 픽셀이 3% 이상이면 → 보임(visible)
    # 3) 둘 다 아니면 → 불확실: conservative하게 '보임'으로 두고 bbox/라벨링 진행
    if closer_ratio >= 0.25:
        if debug:
            print(f"  [Occ] OCCLUDED closer_ratio={closer_ratio:.2f}")
        return False
    
    if near_ratio >= 0.03:
        if debug:
            print(f"  [Occ] VISIBLE near_ratio={near_ratio:.2f}")
        return True
    
    if debug:
        print(f"  [Occ] VISIBLE (fallback) closer={closer_ratio:.2f}, near={near_ratio:.2f}")
    return True


def get_traffic_light_bbox(traffic_light, camera, image_w, image_h, debug=False):
    """
    Project traffic light to 2D bounding box
    
    Priority:
    1. Use level BB from map (accurate per-map size)
    2. Fallback to manual head center projection
    
    Returns:
        bbox: [[min_x, min_y], [max_x, max_y]] or None if not visible
    """
    # Get camera transform
    camera_transform = camera.get_transform()
    camera_location = camera_transform.location
    
    # Get traffic light location
    tl_transform = traffic_light.get_transform()
    tl_location = tl_transform.location
    
    # Check if traffic light is in front of camera
    tl_vector = tl_location - camera_location
    camera_forward = camera_transform.get_forward_vector()
    
    dot_product = (tl_vector.x * camera_forward.x + 
                   tl_vector.y * camera_forward.y + 
                   tl_vector.z * camera_forward.z)
    
    if debug:
        print(f"      [BBOX Debug] dot_product={dot_product:.2f}")
    
    if dot_product <= 0:
        if debug:
            print(f"      [BBOX] Behind camera")
        return None  # Behind camera
    
    # Get distance
    distance = camera_location.distance(tl_location)
    
    if debug:
        print(f"      [BBOX Debug] distance={distance:.1f}m (max={MAX_DIST_TRAFFIC_LIGHT}m)")
    
    if distance > MAX_DIST_TRAFFIC_LIGHT:
        if debug:
            print(f"      [BBOX] Too far")
        return None  # Too far
    
    # ========== 우선순위 1: Level BB 기반 bbox (맵 실제 크기 자동 반영) ==========
    if TL_LEVEL_BBS:  # Level BB가 로드되어 있으면
        bb2d = tl_bbox_from_levelbb(traffic_light, camera, image_w, image_h, 
                                     head_only=True, 
                                     head_height_m=1.0,
                                     head_top_margin_m=0.05)
        if bb2d is not None:
            if debug:
                print(f"      [BBOX] ✓ VALID (from Level BB)")
            return bb2d
        elif debug:
            print(f"      [BBOX] Level BB failed, trying fallback...")
    
    # ========== 백업: 기존 헤드 중심 + 가상 박스 방식 ========== 
    
    # 신호등 헤드(전등 부분) 중심으로 bbox 생성
    # 기둥 하단(Transform 기준)이 아닌 헤드 위치로 중심을 올림
    center_local = carla.Location(0.0, 0.0, TL_HEAD_CENTER_Z)
    
    # 8개 버텍스를 헤드 중심 주변으로 생성
    vertices_world = []
    for sx in (-TL_EXTENT_X, TL_EXTENT_X):
        for sy in (-TL_EXTENT_Y, TL_EXTENT_Y):
            for sz in (-TL_EXTENT_Z, TL_EXTENT_Z):
                v_local = center_local + carla.Location(x=sx, y=sy, z=sz)
                world_point = tl_transform.transform(v_local)
                vertices_world.append(world_point)
    
    if debug:
        head_center_world = tl_transform.transform(center_local)
        print(f"      [BBOX Debug] Head center: ({head_center_world.x:.1f}, {head_center_world.y:.1f}, {head_center_world.z:.1f})")
    
    # Build projection matrix
    fov = float(camera.attributes['fov'])
    K = build_projection_matrix(image_w, image_h, fov)
    
    # World to camera matrix
    world_2_camera = np.array(camera_transform.get_inverse_matrix())
    
    # Project vertices to 2D
    points_2d = []
    for vertex in vertices_world:
        p = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float32)
        pc = world_2_camera @ p  # sensor 좌표계: X=forward, Y=right, Z=up

        X = pc[0]  # depth(전방)
        Y = pc[1]
        Z = pc[2]

        # 카메라 앞에 있는지(전방 X>0)
        if X <= 0:
            continue

        # 컴퓨터비전 좌표계로 재정렬: u ~ Y/X, v ~ -Z/X
        u = K[0, 0] * (Y / X) + K[0, 2]
        v = K[1, 1] * (-Z / X) + K[1, 2]

        # 여유 마진을 두고 화면 범위 체크
        if -100 < u < image_w + 100 and -100 < v < image_h + 100:
            points_2d.append([u, v, X])
    
    if len(points_2d) < 2:  # 완화: 최소 2개 포인트만 있으면 bbox 생성
        if debug:
            print(f"      [BBOX] Not enough points: {len(points_2d)}/8")
        return None  # Not enough visible points
    
    # Get bounding box from projected points
    points_2d = np.array(points_2d)
    min_x = max(0, np.min(points_2d[:, 0]))
    max_x = min(image_w - 1, np.max(points_2d[:, 0]))
    min_y = max(0, np.min(points_2d[:, 1]))
    max_y = min(image_h - 1, np.max(points_2d[:, 1]))
    
    # Check minimum size
    width = max_x - min_x
    height = max_y - min_y
    
    if debug:
        print(f"      [BBOX] Projected: ({min_x:.0f},{min_y:.0f})-({max_x:.0f},{max_y:.0f}), size={width:.0f}x{height:.0f}px")
    
    if width < MIN_BBOX_WIDTH or height < MIN_BBOX_HEIGHT:
        if debug:
            print(f"      [BBOX] Too small (min {MIN_BBOX_WIDTH}x{MIN_BBOX_HEIGHT})")
        return None  # Too small
    
    if width * height < MIN_BBOX_AREA:
        if debug:
            print(f"      [BBOX] Area too small: {width*height:.0f} < {MIN_BBOX_AREA}")
        return None  # Area too small
    
    if debug:
        print(f"      [BBOX] ✓ VALID!")
    
    return np.array([[min_x, min_y], [max_x, max_y]], dtype=np.float32)


# ---------- Main ----------
# carla_eval_kit에서 주로 사용하는 맵들 (신호등이 많은 맵)
target_maps = ['Town01','Town03','Town04','Town05', 'Town10HD']

