import os
import json
from copy import deepcopy
from collections import deque

import cv2
import carla
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from my_auto_config import MyAutoConfig

from utils.hdmap import HDMap

from ultralytics import YOLO

from team_code_autopilot.utils.fsm.autopilot_fsm import build_vehicle_fsm

def _hfov_to_fx(width_px: int, hfov_deg: float) -> float:
    # Horizontal FoV → focal length (pixels)
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

def _vfov_from_hfov(hfov_deg: float, w: int, h: int) -> float:
    # Derive vertical FoV assuming square pixels
    return math.degrees(2.0 * math.atan((h / w) * math.tan(math.radians(hfov_deg) / 2.0)))

def _cam_intrinsics(w: int, h: int, hfov_deg: float):
    fx = _hfov_to_fx(w, hfov_deg)
    vfov = _vfov_from_hfov(hfov_deg, w, h)
    fy = (h / 2.0) / math.tan(math.radians(vfov) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return fx, fy, cx, cy

def _rpy_deg_to_RxRyRz(roll_deg, pitch_deg, yaw_deg):
    # Roll (x), Pitch (y), Yaw (z) in degrees → rotation matrices
    rx, ry, rz = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0, 0, 1]], dtype=np.float32)
    # Unreal/CARLA 순서: yaw(z) → pitch(y) → roll(x) 적용이 일반적
    return Rx, Ry, Rz

def _ego_to_cam_matrix(cam_pos, cam_rot_rpy_deg):
    # 점을 ego→camera로 보내려면: p_cam = R_inv @ (p_ego - cam_pos)
    Rx, Ry, Rz = _rpy_deg_to_RxRyRz(*cam_rot_rpy_deg)
    R_ego_to_cam = (Rz @ Ry @ Rx)  # camera orientation relative to ego
    R_inv = R_ego_to_cam.T         # inverse for coordinates transform
    t = np.array(cam_pos, dtype=np.float32)
    return R_inv, t

def _project_points_to_image(points_ego_xyz, cam_pos, cam_rot_rpy_deg, w, h, hfov_deg):
    """
    points_ego_xyz: (N,3) in ego coords [x forward, y right, z up]
    returns list of (u,v) ints inside image
    """
    fx, fy, cx, cy = _cam_intrinsics(w, h, hfov_deg)
    R_inv, t = _ego_to_cam_matrix(cam_pos, cam_rot_rpy_deg)

    uv_list = []
    for p in points_ego_xyz:
        p_rel = p - t                     # translate by camera position
        p_cam = R_inv @ p_rel             # rotate into camera axes (X fwd, Y right, Z up)
        X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        if X <= 0.05:                     # behind/too close to the camera plane → skip
            continue
        u = int(cx + fx * (Y / X))        # pinhole: u = cx + fx*(Y/X)
        v = int(cy - fy * (Z / X))        #            v = cy - fy*(Z/X)
        if 0 <= u < w and 0 <= v < h:
            uv_list.append((u, v))
    return uv_list


# =============================================================
# Entry point
# =============================================================
def get_entry_point():
    return 'MyAutopilot'


# =============================================================
# Simple Route Planner (from LBC, trimmed)
# =============================================================
class RoutePlanner(object):
    def __init__(self, min_distance: float, max_distance: float):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.is_last = False

        # CARLA 0.9.10 lat/lon → meters
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])

    def set_route(self, global_plan, gps: bool = False):
        self.route.clear()
        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean
            self.route.append((pos, cmd))

    def run_step(self, gps_xy: np.ndarray):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i - 1][0])
            distance = np.linalg.norm(self.route[i][0] - gps_xy)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False


# =============================================================
# Simple controllers
# =============================================================
class PID:
    def __init__(self, Kp, Ki, Kd, dt, u_min=0.0, u_max=1.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.i = 0.0
        self.prev_e = 0.0
        self.u_min, self.u_max = u_min, u_max

    def step(self, e):
        self.i += e * self.dt
        d = (e - self.prev_e) / self.dt if self.dt > 0 else 0.0
        self.prev_e = e
        u = self.Kp * e + self.Ki * self.i + self.Kd * d
        return max(self.u_min, min(self.u_max, u))


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# =============================================================
# MyAutopilot Agent
# =============================================================
class MyAutopilot(autonomous_agent.AutonomousAgent):
    """
    - Sensors: same placement as config, but we only *use* rgb_front (LiDAR는 사용).
    - Follow global route via RoutePlanner (no learning).
    - Emergency stop: if LiDAR detects obstacle within 5m in front corridor → full brake.
    - Overlay control values on rgb_front and display via OpenCV.
    """

    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False

        # Use independent config (no args.txt / no GlobalConfig inheritance)
        self.config = MyAutoConfig()

        # Route planner window from config
        self._route_planner = RoutePlanner(
            self.config.route_planner_min_distance,
            self.config.route_planner_max_distance
        )

        # Control targets
        self.target_speed = float(self.config.target_speed)
        self.steer_gain = float(self.config.steer_gain)

        # Longitudinal PID using config gains
        self.speed_pid = PID(
            Kp=self.config.pid_Kp,
            Ki=self.config.pid_Ki,
            Kd=self.config.pid_Kd,
            dt=self.config.carla_frame_rate,
            u_min=self.config.pid_u_min,
            u_max=self.config.pid_u_max,
        )

        # LiDAR safety params from config
        self.safety_x_min = self.config.safety_x_min
        self.safety_x_max = self.config.safety_x_max
        self.safety_y_abs = self.config.safety_y_abs
        self.safety_z_min = self.config.safety_z_min
        self.safety_z_max = self.config.safety_z_max

        # HUD
        self.show_window = bool(self.config.show_window)
        self.last_rgb = None


        # HDMap Utils
        self.hdmap = None
        self._global_plan_world = None  
        self._global_route_idx = 0          # index into global_plan_* for "current" route[0]
        self._route_len_prev = None         # track how many waypoints RoutePlanner has in its deque

        # YOLO traffic light detection
        self.use_yolo = False
        self.yolo_model = None
        try:
            # Load trained traffic light model
            model_path = './models/best.pt'
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to('cuda' if self._is_cuda_available() else 'cpu')
            
            # Traffic light class mapping (from training)
            self.traffic_light_class_names = {
                0: 'Red',
                1: 'Yellow',
                2: 'Green',
                3: 'Back'
            }
            self.use_yolo = True
            print(f"[YOLO] Traffic light detection enabled with model: {model_path}")
        except Exception as e:
            print(f"[YOLO] Failed to initialize: {e}")
            self.use_yolo = False

        self._prev_global_wp_world = None
        self._current_global_wp_world = None


        # Traffic light detection state
        self.traffic_lights_detected = []
        self.yolo_update_interval = 3  # Run YOLO every 3 frames
        self.yolo_counter = 0



        # ----------------------------------------------------------
        # FSM: build fsm
        self.fsm, _ = build_vehicle_fsm(start_state="Drive")
        self.fsm_state = self.fsm.state  # for optional HUD/logging

    def _is_cuda_available(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False




    # ---------------------------------------------------------
    # Leaderboard hooks
    # ---------------------------------------------------------
    def sensors(self):
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1], 'yaw': self.config.camera_rot_0[2],
                'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                'id': 'rgb_front'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': self.config.carla_frame_rate,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': self.config.carla_fps,
                'id': 'speed'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': self.config.lidar_pos[0], 'y': self.config.lidar_pos[1], 'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1], 'yaw': self.config.lidar_rot[2],
                'id': 'lidar'
            },
        ]
        return sensors

    def _init_if_needed(self):
        if self.initialized:
            return
        # Convert global plan (lat/lon) to meter XY
        self._route_planner.set_route(self._global_plan, True)

        # ---------------- HDMap init (for visualization only) ----------------
        if self.hdmap is None:
            cam_res = (self.config.camera_width, self.config.camera_height)

            # host/port/role: adjust if you store them in config
            hd_host = "127.0.0.1"
            hd_port = 2000
            hd_role = "hero"

            self.hdmap = HDMap(
                host=hd_host,
                port=hd_port,
                role=hd_role,
                cam_res=cam_res,
                frustum_max_dist=float(self.config.route_planner_max_distance),
                sensor_tick=0.0,
                is_visualize=True,  # show HDMap window for debugging
            )
            
            print("[HDMap] Initialized HDMap utility for visualization.")

        self.initialized = True

    def _get_position_xy(self, gps_latlon):
        gps = gps_latlon  # lat, lon
        gps_xy = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps_xy

    def tick(self, input_data):
        rgb = input_data['rgb_front'][1][:, :, :3]
        self.last_rgb = rgb.copy()

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if np.isnan(compass):
            compass = 0.0
        lidar = input_data['lidar'][1][:, :3]

        # YOLO traffic light detection (every N frames)
        if self.use_yolo:
            self.yolo_counter += 1
            if self.yolo_counter >= self.yolo_update_interval:
                self.traffic_lights_detected = self._detect_traffic_lights(rgb)
                self.yolo_counter = 0


        return {
            'rgb': rgb,
            'gps_ll': gps,
            'speed': speed,
            'compass': compass,
            'lidar': lidar,
            'traffic_lights': self.traffic_lights_detected,

        }
    

    def _route_overlaps_bbox(self, route_pts, bbox: dict, margin: float = 0.0) -> bool:
        """
        Check if a route (polyline) overlaps the ground-projected bbox rectangle.

        Args:
            route_pts:
                - np.ndarray of shape (N, 2) or (N, 3) in world frame
                  (usually one cluster from center_mask_clustered)
            bbox:
                - dict from ga.get_vehicle_bbox, with 'corners_world' etc.
            margin:
                - optional extra expansion [meters] of the bbox footprint
                  (for safety margin / tolerance)

        Returns:
            True if any route point lies inside the rectangle (with margin),
            or any route segment intersects the rectangle edges.
        """
        if route_pts is None or bbox is None:
            return False

        route = np.asarray(route_pts, dtype=float)
        if route.ndim != 2 or route.shape[0] == 0:
            return False

        # Use XY only
        route_xy = route[:, :2]

        # ----- 1) Extract bbox footprint (ground rectangle) -----
        corners_world = bbox.get("corners_world", None)
        if not corners_world:
            # Fallback: use gt_array as a point; not really a rectangle → treat as small circle
            gt = bbox.get("gt_array", None)
            if gt is None:
                return False
            center = np.asarray(gt[:2], dtype=float)
            # Simply check if any route point is within margin of the center
            d2 = np.sum((route_xy - center[None, :]) ** 2, axis=1)
            return bool(np.any(d2 <= (margin ** 2 if margin > 0 else 0.5 ** 2)))

        corners = np.asarray(corners_world, dtype=float)  # (8, 3) typically
        foot_xy = corners[:, :2]  # project to ground

        # Deduplicate XY pairs (bottom & top corners share XY)
        uniq_xy, uniq_idx = np.unique(np.round(foot_xy, 3), axis=0, return_index=True)
        poly_xy = foot_xy[uniq_idx]

        if poly_xy.shape[0] < 3:
            return False  # not enough to define polygon

        # ----- 2) Order rectangle vertices (clockwise) -----
        centroid = poly_xy.mean(axis=0)
        rel = poly_xy - centroid[None, :]
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        order = np.argsort(angles)
        poly = poly_xy[order]  # (M, 2), M ~ 4

        # Optional: expand polygon by 'margin' using simple axis-aligned margin
        # (approximation; good enough for safety check)
        if margin > 0.0:
            min_x, min_y = poly[:, 0].min() - margin, poly[:, 1].min() - margin
            max_x, max_y = poly[:, 0].max() + margin, poly[:, 1].max() + margin
        else:
            min_x, min_y = poly[:, 0].min(), poly[:, 1].min()
            max_x, max_y = poly[:, 0].max(), poly[:, 1].max()

        # ----- 3) Fast AABB rejection between route and bbox -----
        r_min_x, r_min_y = route_xy[:, 0].min(), route_xy[:, 1].min()
        r_max_x, r_max_y = route_xy[:, 0].max(), route_xy[:, 1].max()

        if (r_max_x < min_x or r_min_x > max_x or
            r_max_y < min_y or r_min_y > max_y):
            return False  # clearly no overlap

        # ----- 4) Precise point-in-polygon test (convex poly) -----
        def point_in_convex_poly(p, poly_pts):
            """
            Check if point p is inside convex polygon poly_pts (2D).
            Uses sign of cross products along edges.
            """
            x, y = p
            sign = None
            for i in range(len(poly_pts)):
                x1, y1 = poly_pts[i]
                x2, y2 = poly_pts[(i + 1) % len(poly_pts)]
                cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                if cross == 0:
                    continue  # on the edge is OK → treat as inside
                curr_sign = cross > 0
                if sign is None:
                    sign = curr_sign
                elif sign != curr_sign:
                    return False  # different side → outside
            return True

        # If any route point lies inside the polygon → overlapping
        for p in route_xy:
            if point_in_convex_poly(p, poly):
                return True

        # ----- 5) Segment vs edge intersection check -----
        def segments_intersect(p1, p2, q1, q2):
            """
            Check if line segments p1-p2 and q1-q2 intersect in 2D.
            """
            def orient(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

            o1 = orient(p1, p2, q1)
            o2 = orient(p1, p2, q2)
            o3 = orient(q1, q2, p1)
            o4 = orient(q1, q2, p2)

            # General case
            if (o1 * o2 < 0) and (o3 * o4 < 0):
                return True

            # Collinear / endpoint cases can be added if needed; for now we ignore
            return False

        # Build polygon edges
        edges = []
        M = len(poly)
        for i in range(M):
            q1 = poly[i]
            q2 = poly[(i + 1) % M]
            edges.append((q1, q2))

        # Check each route segment vs each polygon edge
        for i in range(len(route_xy) - 1):
            p1 = route_xy[i]
            p2 = route_xy[i + 1]
            for q1, q2 in edges:
                if segments_intersect(p1, p2, q1, q2):
                    return True

        return False

    
    def _detect_traffic_lights(self, rgb_image):
        """
        Detect traffic lights using trained YOLO model
        
        Args:
            rgb_image: (H, W, 3) numpy array in RGB format
            
        Returns:
            list of dicts: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'class': int, 'class_name': str}, ...]
        """
        if not self.use_yolo or self.yolo_model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(rgb_image, verbose=False, imgsz=640)
            
            detections = []
            for r in results:
                boxes = r.boxes  # Detections object
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter by confidence threshold
                    if conf > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_name = self.traffic_light_class_names.get(cls, 'Unknown')
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf,
                            'class': cls,
                            'class_name': class_name
                        })
            
            return detections
        
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return []    

    def run_step(self, input_data, timestamp):
        self.step += 1
        self._init_if_needed()

        data = self.tick(input_data)

        # Position in meters (XY)
        pos_xy = self._get_position_xy(data['gps_ll'])

        # Route planning: obtain next waypoint & command
        route = self._route_planner.run_step(pos_xy)
        next_wp, next_cmd = route[1] if len(route) > 1 else route[0]

        # next_wp_world = np.array([float(next_wp[0]), float(next_wp[1]), 0.0], dtype=float)
        next_wp_world = np.array([float(next_wp[1]), -float(next_wp[0]), 0.0], dtype=float)


        # 3) If this is the first time, just initialize current and skip A*
        if self._current_global_wp_world is None:
            self._current_global_wp_world = next_wp_world
        else:
            # Check if next_wp changed compared to current_global_wp
            if not np.allclose(
                next_wp_world[:2],
                self._current_global_wp_world[:2],
                atol=1e-3
            ):
                # Waypoint changed → shift current → prev, update current
                self._prev_global_wp_world = self._current_global_wp_world
                self._current_global_wp_world = next_wp_world

        # # -----------------------------------------------------
        # # HDMap debug: update masks + A* route (for visualization only)
        # # -----------------------------------------------------
        # if self.hdmap is not None:
        #     # 1) Update HDMap internal state (ego pose, masks, etc.)
        #     self.hdmap.tick()
        #     self.hdmap.update_route_between_globals(self._prev_global_wp_world, self._current_global_wp_world)
        # else:
        #     print("[HDMap] Warning: HDMap utility not initialized; skipping HDMap debug update.")

        # -----------------------------------------------------
        # 3) HDMap update + local centerline waypoint
        # -----------------------------------------------------
        hd_wp_xy = None  # waypoint in [north, east]

        if self.hdmap is not None:
            # Update HDMap internals (ego pose, masks, etc.)
            self.hdmap.tick()

            # Update local route between prev and current globals
            self.hdmap.update_route_between_globals(
                self._prev_global_wp_world,
                self._current_global_wp_world,
            )

            # Get next mid-lane waypoint in world frame [x, y, z]
            hd_wp_world = self.hdmap.get_next_waypoint()

            if hd_wp_world is not None:
                hd_wp_world = np.asarray(hd_wp_world, dtype=float)

                # Convert world [x, y] back to "map" [north, east]
                # world_x = east, world_y = -north → north = -world_y, east = world_x
                hd_north = -hd_wp_world[1]
                hd_east = hd_wp_world[0]
                hd_wp_xy = (float(hd_north), float(hd_east))
        else:
            print(
                "[HDMap] Warning: HDMap not initialized; "
                "skipping HDMap update and mid-lane waypoint."
            )

        # -----------------------------------------------------
        # 4) Choose waypoint for control (HDMap first, fallback: global route)
        # -----------------------------------------------------
        if hd_wp_xy is not None:
            target_n, target_e = hd_wp_xy
        else:
            target_n, target_e = float(next_wp[0]), float(next_wp[1])



        # # Transform next waypoint to ego local frame using compass
        # dn = float(next_wp[0] - pos_xy[0])   # northing (lat)
        # de = float(next_wp[1] - pos_xy[1])   # easting  (lon)

        # Transform chosen waypoint to ego-local frame using compass
        dn = float(target_n - pos_xy[0])  # northing delta
        de = float(target_e - pos_xy[1])  # easting  delta

        # --- CARLA compass: 0=N, +CW  →  yaw_math: 0=East, +CCW ---
        bearing = float(data['compass'])     # radians
        yaw = np.pi/2.0 - bearing            # world-yaw from +x(East), CCW

        # --- world([east, north]) → ego([x_forward, y_left]) ---
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[ c,  s],
                    [-s,  c]])
        vec_local = R @ np.array([de, dn], dtype=np.float32)  # [x_forward, y_left]

        v = float(data['speed'])  # current speed m/s


        # -----------------------------------------------------
        # LiDAR emergency stop (AABB in ego coordinates)
        # CARLA LiDAR: x forward, y right, z up
        # -----------------------------------------------------
        lidar = data['lidar'][:, :3].copy()

        # 1) 센서 yaw 보정 (lidar_rot[2] = -90° → 차량 전방 정렬)
        yaw_deg = float(self.config.lidar_rot[2])
        yaw_rad = math.radians(yaw_deg)
        cy, sy = np.cos(-yaw_rad), np.sin(-yaw_rad)   # -yaw로 회전 보정
        R_align = np.array([[cy, -sy],
                            [sy,  cy]], dtype=np.float32)

        xy_ego = lidar[:, :2] @ R_align.T
        x_forward = xy_ego[:, 0]    # 차량 전방
        y_right  = xy_ego[:, 1]    # 차량 오른쪽
        z_up     = lidar[:, 2]     # 위쪽

        # 2) 안전박스 조건 (앞으로 5m, 좌우 ±safety_y_abs, 높이 safety_z_min~safety_z_max)
        mask = (
            (x_forward > self.safety_x_min) & (x_forward < self.safety_x_max) &
            (np.abs(y_right) < self.safety_y_abs) &
            (z_up > self.safety_z_min) & (z_up < self.safety_z_max)
        )
        obstacle = bool(np.any(mask))




        # -----------------------------------------------------
        # Traffic Light Detection
        # -----------------------------------------------------

        # Traffic light-based speed adjustment with color detection
        traffic_lights = data.get('traffic_lights', [])
        red_light_detected = False
        yellow_light_detected = False

        if len(traffic_lights) > 0 and self.use_yolo:
            # Find largest (closest) traffic light
            largest_tl = max(traffic_lights, key=lambda t: (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1]))
            bbox_area = (largest_tl['bbox'][2] - largest_tl['bbox'][0]) * (largest_tl['bbox'][3] - largest_tl['bbox'][1])
            class_name = largest_tl['class_name']
            
            # Only react if traffic light is close enough (large bbox)
            if bbox_area > 5000:
                if class_name == 'Red':
                    red_light_detected = True
                    self.target_speed = 0.0  # Full stop for red light
                    print(f"[Traffic Light] RED - Stopping (area: {bbox_area:.0f})")
                elif class_name == 'Yellow':
                    yellow_light_detected = True
                    self.target_speed = min(self.target_speed, 2.0)  # Slow down for yellow
                    print(f"[Traffic Light] YELLOW - Slowing (area: {bbox_area:.0f})")
                elif class_name == 'Green':
                    # Green light - restore normal speed
                    default_speed = float(self.config.target_speed)
                    self.target_speed = default_speed
                    print(f"[Traffic Light] GREEN - Go (area: {bbox_area:.0f})")
        

 
        # -----------------------------------------------------
        # FSM step
        # -----------------------------------------------------
        # cargo = {
        #     "obstacle": obstacle,
        #     # Hook up traffic light later if you have it:
        #     "red": False,
        #     "dt": float(self.speed_pid.dt),
        #     "speed": v,
        #     "timestamp": float(timestamp) if timestamp is not None else float(self.step),
        #     # You can add planner info if needed:
        #     # "planner_cmd": str(next_cmd),
        # }
        # fsm_state, fsm_info = self.fsm.step(cargo)
        # self.fsm_state = fsm_state  # for optional HUD/logging
        
        cargo = {
            # Time
            "dt": float(self.speed_pid.dt),
            "t": float(timestamp) if timestamp is not None else None,

            # Ego state
            "speed": float(v),

            # Gating signals (for now: only obstacle is wired;
            # everything else is "no restriction" → always DRIVE)
            "obstacle_ahead": bool(obstacle),
            "obstacle_distance": None,          # not used yet

            "tl_red": False,
            "tl_near_stopline": False,

            "stop_sign_ahead": False,
            "ss_near_stopline": False,

            # Desired free-flow speed in DRIVE
            "cruise_target": float(self.target_speed),

            # Optional: FSM can carry a waypoint if you want.
            # For now we let the HDMap / earlier code pick the target
            # and only use vec_local for lateral control.
            "waypoint": None,
        }

        fsm_state, plan = self.fsm.step(cargo)
        self.fsm_state = fsm_state  # for HUD/logging


        steer, throttle_cmd, brake_cmd = 0.0, 0.0, 0.0

        if fsm_state == "Drive":
            # -----------------------------------------------------
            # Lateral control: simple pure-pursuit / heading error
            # -----------------------------------------------------
            kx, ky = float(vec_local[0]), float(vec_local[1])
            steer_angle = math.atan2(ky, max(1e-3, kx))      # [-pi, pi]
            # ky: +면 왼쪽 → CARLA steer는 왼쪽이 음수이므로 부호 반전
            steer = clip(-self.steer_gain * (steer_angle / (math.pi / 2.0)), -1.0, 1.0)

            # -----------------------------------------------------
            # Longitudinal PID for target speed
            # -----------------------------------------------------
            v = float(data['speed'])
            e_v = self.target_speed - v
            throttle_cmd = self.speed_pid.step(e_v)
            brake_cmd = 0.0

        elif fsm_state == "Stop":
            throttle_cmd = 0.0
            brake_cmd = 1.0
        else:
            # Safe fallback if a new/unknown state appears
            throttle_cmd = 0.0
            brake_cmd = 1.0




        # -----------------------------------------------------
        # Compose VehicleControl
        # -----------------------------------------------------
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle_cmd)
        control.brake = float(brake_cmd)

        # -----------------------------------------------------
        # On-screen HUD overlay via OpenCV
        # -----------------------------------------------------
        try:
            if self.show_window and self.last_rgb is not None:
                hud = self.last_rgb.copy()
                h, w, _ = hud.shape

                # ----- 텍스트 HUD -----
                txts = [
                    f"step: {self.step}",
                    f"cmd: {getattr(next_cmd, 'name', str(next_cmd))}",
                    f"speed: {v:.2f} m/s (target {self.target_speed:.1f})",
                    f"steer: {control.steer:+.3f}",
                    f"throttle: {control.throttle:.3f}",
                    f"brake: {control.brake:.3f}",
                    f"front_obstacle: {obstacle}",
                    f"fsm_state: {getattr(self, 'fsm_state', 'N/A')}",   # ★ NEW
                ]
                
                pos_xy = self._get_position_xy(data['gps_ll'])
                bearing = float(data['compass'])
                yaw = math.pi/2.0 - bearing
                c, s = math.cos(yaw), math.sin(yaw)
                R_world_to_ego = np.array([[ c,  s],
                                        [-s,  c]], dtype=np.float32)

                dn_next = float(next_wp[0] - pos_xy[0])
                de_next = float(next_wp[1] - pos_xy[1])
                x_fwd_next, y_left_next = (R_world_to_ego @ np.array([de_next, dn_next], dtype=np.float32)).tolist()

                txts.append(f"next_wp_local: x_fwd={x_fwd_next:.1f}m, y_left={y_left_next:.1f}m")
                
                y0 = 28
                for i, t in enumerate(txts):
                    cv2.putText(hud, t, (12, y0 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


                # # ====== YOLO Traffic Light Bounding Boxes ======
                # traffic_lights = data.get('traffic_lights', [])
                # if self.use_yolo and len(traffic_lights) > 0:
                #     for tl in traffic_lights:
                #         x1, y1, x2, y2 = tl['bbox']
                #         conf = tl['conf']
                #         class_name = tl['class_name']
                        
                #         # Color-coded bounding boxes based on traffic light state
                #         if class_name == 'Red':
                #             color = (0, 0, 255)  # Red in BGR
                #         elif class_name == 'Yellow':
                #             color = (0, 255, 255)  # Yellow in BGR
                #         elif class_name == 'Green':
                #             color = (0, 255, 0)  # Green in BGR
                #         else:  # Back (off)
                #             color = (128, 128, 128)  # Gray in BGR
                        
                #         # Draw bounding box
                #         cv2.rectangle(hud, (x1, y1), (x2, y2), color, 3)
                        
                #         # Draw label with confidence
                #         label = f"{class_name}: {conf:.2f}"
                #         label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                #         cv2.rectangle(hud, (x1, y1 - label_size[1] - 8), 
                #                      (x1 + label_size[0], y1), color, -1)
                #         cv2.putText(hud, label, (x1, y1 - 5),
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                #     # Display detection count
                #     tl_count_text = f"Traffic Lights: {len(traffic_lights)}"
                #     cv2.putText(hud, tl_count_text, (12, h - 20),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)





                # ====== Waypoints → Camera Projection (next 5) ======
                # Camera intrinsics from horizontal FOV
                hfov = float(self.config.camera_fov)                 # deg
                fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
                # derive vertical FOV assuming square pixels
                vfov = 2.0 * math.degrees(math.atan((h / w) * math.tan(math.radians(hfov) / 2.0)))
                fy = (h / 2.0) / math.tan(math.radians(vfov) / 2.0)
                cx, cy = w / 2.0, h / 2.0

                # World(N/E) → Ego(x_fwd, y_left) rotation from compass
                pos_xy = self._get_position_xy(data['gps_ll'])      # [lat_m, lon_m] = [north, east]
                bearing = float(data['compass'])                    # rad, 0=N, +CW
                yaw = math.pi/2.0 - bearing                         # 0=E, +CCW
                c, s = math.cos(yaw), math.sin(yaw)
                R_world_to_ego = np.array([[ c,  s],
                                        [-s,  c]], dtype=np.float32)

                # Camera extrinsics (ego→cam): p_cam = R_inv @ (p_ego - t)
                cam_pos = np.array(self.config.camera_pos, dtype=np.float32)       # [x,y,z] in ego
                roll, pitch, yaw_deg = self.config.camera_rot_0                    # deg
                rx, ry, rz = map(math.radians, (roll, pitch, yaw_deg))
                cxr, sxr = math.cos(rx), math.sin(rx)
                cyr, syr = math.cos(ry), math.sin(ry)
                czr, szr = math.cos(rz), math.sin(rz)
                Rx = np.array([[1, 0, 0],
                            [0, cxr, -sxr],
                            [0, sxr,  cxr]], dtype=np.float32)
                Ry = np.array([[ cyr, 0, syr],
                            [ 0,  1,  0 ],
                            [-syr, 0, cyr]], dtype=np.float32)
                Rz = np.array([[ czr, -szr, 0],
                            [ szr,  czr, 0],
                            [  0,    0,  1]], dtype=np.float32)
                R_ego_to_cam = (Rz @ Ry @ Rx)     # Unreal/CARLA: yaw→pitch→roll
                R_inv = R_ego_to_cam.T

                # Collect next 5 waypoints (current included)
                waypoints = list(route)[:6]
                pts_ego = []
                for j, (wp, cmd) in enumerate(waypoints):
                    dn = float(wp[0] - pos_xy[0])         # north (lat)
                    de = float(wp[1] - pos_xy[1])         # east  (lon)
                    x_fwd, y_left = (R_world_to_ego @ np.array([de, dn], dtype=np.float32)).tolist()
                    y_right = -y_left
                    z_up = 0.0                             # assume ground plane for viz
                    pts_ego.append(np.array([x_fwd, y_right, z_up], dtype=np.float32))

                # Project to image and draw
                for j, p in enumerate(pts_ego):
                    p_cam = R_inv @ (p - cam_pos)          # ego→cam
                    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
                    if X <= 0.05:                          # behind/too close → skip
                        continue
                    u = int(cx + fx * (Y / X))
                    v = int(cy - fy * (Z / X))
                    if 0 <= u < w and 0 <= v < h:
                        color = (0, 0, 255) if j == 0 else (255, 0, 0)
                        cv2.circle(hud, (u, v), 6, color, -1)

                hud = cv2.resize(hud, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                # hud = cv2.resize(hud, (w, h), interpolation=cv2.INTER_LINEAR)

                cv2.imshow('rgb_front', hud)
                cv2.waitKey(1)

        except Exception:
            # If running headless, just ignore GUI errors
            pass

        return control

    def destroy(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
