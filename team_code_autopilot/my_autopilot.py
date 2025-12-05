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

from team_code_autopilot.utils.hdmap import HDMap

from utils.traffic_light_detector import TrafficLightDetector

from team_code_autopilot.fsm.autopilot_fsm import (
    build_vehicle_fsm,
    build_cargo_from_hdmap,
    PlannerOutput,
)


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
    CARLA leaderboard agent wiring together:

    - RoutePlanner for a global route in (north, east) space.
    - HDMap for local lane geometry (used by the behavioural FSM).
    - TrafficLightDetector (YOLO + LiDAR) for:
        * whether a traffic light is red
        * whether a stop sign is present
        * distance to the relevant traffic light
        * distance to the relevant stop sign
    - autopilot_fsm.FSM as the ONLY component that decides DRIVE/STOP and
      produces a PlannerOutput: (waypoint, target_speed, mode, reason).
    - A PID-based low-level controller that:
        * follows PlannerOutput.waypoint for steering, and
        * tracks PlannerOutput.target_speed for throttle/brake.

    my_autopilot.py only:
        - sets up sensors, vehicle and HUD,
        - runs LiDAR safety-box and YOLO-based detection,
        - feeds those signals plus HDMap/ego state into autopilot_fsm.build_cargo_from_hdmap,
        - runs the FSM and applies its PlannerOutput.

    It NEVER touches or augments the FSM cargo dict and does NOT implement
    its own behavioural logic or fallback path. Behaviour (drive vs stop)
    is entirely decided in autopilot_fsm.py.
    """

    def setup(self, path_to_conf_file, route_index=None):
        print("[MyAutopilot] SETUP CALLED - HUD DEBUG VERSION")
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
        self.show_window = True
        print(f"[MyAutopilot] show_window = {self.show_window}")
        self.last_rgb = None



        # HDMap Utils
        self.hdmap = None
        self._global_plan_world = None  
        self._global_route_idx = 0          # index into global_plan_* for "current" route[0]
        self._route_len_prev = None         # track how many waypoints RoutePlanner has in its deque


        # Traffic light & stop-sign detection using the shared module
        self.traffic_light_detector = TrafficLightDetector(
            model_path='./models/Traffic_GT.pt',  # same model as TL+SS agent
            use_cuda=True,
            config=self.config,
        )

        self.use_yolo = self.traffic_light_detector.use_yolo

        # Traffic light detection state
        self.traffic_lights_detected = []
        self.yolo_update_interval = 1  # run YOLO every frame (or tune if needed)
        self.yolo_counter = 0


        self._prev_global_wp_world = None
        self._current_global_wp_world = None

        # ----------------------------------------------------------
        # FSM: build fsm
        # ----------------------------------------------------------
        self.fsm, self.fsm_cfg = build_vehicle_fsm(start_state="Drive")
        self.fsm_state = self.fsm.state  # for optional HUD/logging


        # Detour obstacle-relaxation bookkeeping
        self._detour_enter_time_s = None          # timestamp when we entered Detour
        self._detour_elapsed_s = 0.0              # local accumulator if timestamp is None
        # Must match the default in build_cargo_from_hdmap
        self._obstacle_check_base = 30.0          # base obstacle check distance in meters
        self._detour_relax_factor = float(self.fsm_cfg.detour_relax_obstacle_factor)
        self._detour_relax_duration_s = float(self.fsm_cfg.detour_relax_obstacle_duration_s)



    def _is_cuda_available(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def set_global_plan(self, global_plan_gps, global_plan_world):
        """
        Called by the leaderboard to provide the full route.

        - global_plan_gps: list of (gps_dict, command) where gps_dict has 'lat', 'lon'
        - global_plan_world: list of (transform, command) in world coordinates

        We store the GPS plan for the RoutePlanner and keep the world plan
        only for potential debugging/visualization.
        """
        self._global_plan = global_plan_gps
        self._global_plan_world = global_plan_world


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
        if self.traffic_light_detector.use_yolo:
            self.yolo_counter += 1
            if self.yolo_counter >= self.yolo_update_interval:
                self.traffic_lights_detected = self.traffic_light_detector.detect_traffic_lights(rgb)
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

    
    def run_step(self, input_data, timestamp):
        """
        Main CARLA agent tick.

        - Reads sensors via self.tick(...)
        - Updates global route and HDMap (administrative)
        - Computes safety box obstacle & high-level TL/stop-signf signals
        - Builds cargo via autopilot_fsm.build_cargo_from_hdmap (ONLY place cargo is built)
        - Steps FSM to get PlannerOutput (waypoint + target_speed + state/reason)
        - Runs low-level PID control that follows the FSM waypoint & target speed

        NOTE: This method does *not* construct or modify cargo on its own beyond
        calling build_cargo_from_hdmap, and it does *not* compute any fallback
        path. If the FSM does not provide a waypoint, we simply stop.
        """
        
        self.step += 1
        self._init_if_needed()

        data = self.tick(input_data)

        # -----------------------------------------------------
        # 1) Global route → next sparse waypoint
        # -----------------------------------------------------
        # Position in meters (XY)
        pos_xy = self._get_position_xy(data['gps_ll'])

        # Route planning: obtain next waypoint & command in (north, east)
        route = self._route_planner.run_step(pos_xy)

        # If for some reason there is no route left, stop safely
        if len(route) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control
        
        next_wp, next_cmd = route[1] if len(route) > 1 else route[0]

        # Convert RoutePlanner waypoint to world frame [x, y, z]
        # Route planner: (north, east)
        # World: x = east, y = -north
        next_wp_world = np.array(
            [float(next_wp[1]), -float(next_wp[0]), 0.0],
            dtype=float,
        )


        # -----------------------------------------------------
        # 2) Maintain "prev / current" global waypoints for HDMap
        # -----------------------------------------------------
        if self._prev_global_wp_world is None or self._current_global_wp_world is None:
            # First time initialization
            self._current_global_wp_world = next_wp_world
            self._prev_global_wp_world = next_wp_world

        # If the sparse global waypoint changed, update the pair
        if not np.allclose(
            next_wp_world[:2],
            self._current_global_wp_world[:2],
            atol=1e-3,
        ):
            self._prev_global_wp_world = self._current_global_wp_world
            self._current_global_wp_world = next_wp_world


        # -----------------------------------------------------
        # 3) HDMap update (ego + local path), but *no* direct control from here
        # -----------------------------------------------------
        if self.hdmap is not None:
            # Update HDMap internals (ego pose, masks, actor snapshots, etc.)
            self.hdmap.tick()

            # Update local route between prev and current globals
            # (autopilot_fsm.build_cargo_from_hdmap will call hdmap.get_next_waypoint())
            self.hdmap.update_route_between_globals(
                self._prev_global_wp_world,
                self._current_global_wp_world,
            )
        else:
            print(
                "[HDMap] Warning: HDMap not initialized; "
                "skipping HDMap update."
            )



        # -----------------------------------------------------
        # 4) LiDAR safety box → obstacle boolean - NOT USED RIGHT NOW -- must be changed so that lidar is projected onto hdmap
        # -----------------------------------------------------
        # lidar = data['lidar']  # (N, 3): [x_forward, y_right, z_up] in ego frame

        # x_fwd = lidar[:, 0]
        # y_right = lidar[:, 1]
        # z_up = lidar[:, 2]

        # # Safety corridor in ego frame
        # mask = (
        #     (x_fwd > self.safety_x_min) & (x_fwd < self.safety_x_max) &
        #     (np.abs(y_right) < self.safety_y_abs) &
        #     (z_up > self.safety_z_min) & (z_up < self.safety_z_max)
        # )
        # obstacle = bool(np.any(mask))


        # -----------------------------------------------------
        # 5) Traffic light & stop-sign high-level signals
        # -----------------------------------------------------
        # traffic_lights = data.get("traffic_lights", [])

        # tl_red = False
        # stop_sign_present = False
        # tl_distance = None
        # stop_sign_distance = None

        # if self.traffic_light_detector.use_yolo and len(traffic_lights) > 0:
        #     tl_red, stop_sign_present, tl_distance, stop_sign_distance = (
        #         self.traffic_light_detector.get_high_level_signals(
        #             detections=traffic_lights,
        #             lidar_points=data["lidar"],
        #             rgb_image=self.last_rgb,
        #             cam_pos=self.config.camera_pos,
        #             cam_rot_rpy_deg=self.config.camera_rot_0,
        #             lidar_pos=self.config.lidar_pos,
        #             lidar_rot_rpy_deg=self.config.lidar_rot,
        #         )
        #     )

        # NOTE:
        # We no longer adjust self.target_speed directly based on traffic lights;
        # instead, these 4 high-level signals go into the behavioural FSM via
        # build_cargo_from_hdmap, and the FSM decides whether we are in DRIVE or STOP.

        
        # -----------------------------------------------------
        # 6) FSM cargo building (HDMap + ego + perception)
        # -----------------------------------------------------

        # -----------------------------------------------------
        # 6) FSM cargo building (HDMap + ego + perception)
        # -----------------------------------------------------

        # Choose dynamic obstacle lookahead distance for the FSM's obstacle_ahead flag.
        base_obs_dist = getattr(self, "_obstacle_check_base", 30.0)
        obstacle_distance = base_obs_dist

        # If we are currently in Detour, keep the obstacle distance reduced
        # for a fixed window after entering Detour, then snap back to normal.
        if self.fsm.state == "Detour" and self._detour_enter_time_s is not None:

            # Prefer the absolute leaderboard timestamp if available
            if timestamp is not None:
                dt_detour = float(timestamp) - float(self._detour_enter_time_s)
            else:
                # Fallback: integrate the elapsed time using the controller dt
                self._detour_elapsed_s = float(getattr(self, "_detour_elapsed_s", 0.0)) + float(self.speed_pid.dt)
                dt_detour = self._detour_elapsed_s

            dur = float(self._detour_relax_duration_s)
            relax_f = float(self._detour_relax_factor)

            if dur <= 0.0 or dt_detour <= dur:
                # During the configured duration (or if dur <= 0),
                # use the fully relaxed distance.
                obstacle_distance = base_obs_dist * relax_f
            else:
                # After the window, use the normal base distance again.
                obstacle_distance = base_obs_dist




        # All cargo construction is centralized in autopilot_fsm.build_cargo_from_hdmap.
        # my_autopilot.py only supplies raw inputs (HDMap, ego, LiDAR, TL/SS signals).
        cargo = build_cargo_from_hdmap(
            hdmap_obj=self.hdmap,
            ego_actor=getattr(self.hdmap, "ego", None) if self.hdmap is not None else None,
            dt=float(self.speed_pid.dt),
            t=float(timestamp) if timestamp is not None else None,
            cruise_target=float(self.target_speed),
            obstacle_distance=obstacle_distance,  

            # just for testing
            tl_red_from_vision=False,
            tl_distance=None,
            stop_sign_ahead_from_vision=False,
            stop_sign_distance=None,

            # tl_red_from_vision=tl_red,
            # tl_distance=tl_distance,s
            # stop_sign_ahead_from_vision=stop_sign_present,
            # stop_sign_distance=stop_sign_distance,
        )

        # IMPORTANT: do *not* modify cargo after this point.
        # All logic about "obstacle_ahead", "tl_red", "tl_near_stopline",
        # "stop_sign_ahead", "ss_near_stopline" lives inside autopilot_fsm.py.


        # -----------------------------------------------------
        # 7) FSM step → state + high-level plan
        # -----------------------------------------------------
        prev_state = self.fsm_state
        fsm_state, plan = self.fsm.step(cargo)
        self.fsm_state = fsm_state                    # for HUD/logging
        self.fsm_plan = plan                          # store whole PlannerOutput for HUD
        self.fsm_reason = getattr(plan, "reason", "") # "obstacle", "red_light", ...

        # Detour entry/exit timing for obstacle-distance relaxation
        if fsm_state == "Detour":
            # First time entering Detour in this episode
            if prev_state != "Detour" or self._detour_enter_time_s is None:
                self._detour_enter_time_s = float(timestamp) if timestamp is not None else 0.0
                self._detour_elapsed_s = 0.0  # reset local elapsed-time counter
        else:
            # Left Detour → clear timers
            self._detour_enter_time_s = None
            self._detour_elapsed_s = 0.0






        # -----------------------------------------------------
        # 8) Low-level control using FSM outputs
        # -----------------------------------------------------
        steer, throttle_cmd, brake_cmd = 0.0, 0.0, 0.0

        # a) Waypoint for lateral control comes *only* from the FSM
        has_valid_waypoint = bool(plan is not None and plan.waypoint is not None)
        target_n = None
        target_e = None
        v = float(data['speed'])

        if has_valid_waypoint:
            wp_world = np.asarray(plan.waypoint, dtype=float).reshape(-1)
            # HDMap world convention: x = East, y = -North
            # Route planner convention: (north, east)
            target_n = -wp_world[1]
            target_e = wp_world[0]

            # Transform FSM waypoint to ego-local frame using compass
            dn = float(target_n - pos_xy[0])  # northing delta
            de = float(target_e - pos_xy[1])  # easting  delta

            bearing = float(data['compass'])      # rad, 0=N, +CW
            yaw = np.pi / 2.0 - bearing           # world-yaw from +x(East), CCW
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c,  s],
                          [-s, c]], dtype=np.float32)
            vec_local = R @ np.array([de, dn], dtype=np.float32)  # [x_forward, y_left]
        else:
            # No valid waypoint from FSM → we do NOT invent one.
            # Keep vec_local at zero; we will only allow STOP behavior.
            vec_local = np.array([0.0, 0.0], dtype=np.float32)

        # b) Longitudinal target speed from FSM plan
        plan_target_speed = (
            float(plan.target_speed)
            if (plan is not None and plan.target_speed is not None)
            else float(self.target_speed)
        )

        # c) State-dependent control
        if fsm_state in ("Drive", "Detour") and has_valid_waypoint:
            # Lateral control: simple heading error from vec_local
            kx, ky = float(vec_local[0]), float(vec_local[1])
            steer_angle = math.atan2(ky, max(1e-3, kx))  # [-pi, pi]
            # CARLA: left turn is negative steer
            steer = clip(-self.steer_gain * (steer_angle / (math.pi / 2.0)), -1.0, 1.0)

            # Longitudinal PID towards FSM target speed
            e_v = plan_target_speed - v
            throttle_cmd = self.speed_pid.step(e_v)
            brake_cmd = 0.0

        elif (
            fsm_state == "Stop"
            or (fsm_state in ("Drive", "Detour") and not has_valid_waypoint)
        ):
            # STOP: explicit STOP state, or we refuse to drive without an FSM waypoint
            steer = 0.0
            throttle_cmd = 0.0
            brake_cmd = 1.0

        else:
            # Safe fallback for unknown states
            steer = 0.0
            throttle_cmd = 0.0
            brake_cmd = 1.0




        # -----------------------------------------------------
        # 9) Compose VehicleControl
        # -----------------------------------------------------
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle_cmd)
        control.brake = float(brake_cmd)


        # -----------------------------------------------------
        # 10) On-screen HUD overlay via OpenCV
        # -----------------------------------------------------
        try:
            if self.show_window and self.last_rgb is not None:
                hud = self.last_rgb.copy()
                h, w, _ = hud.shape

                # ----- Text HUD -----
                fsm_state_str = getattr(self, "fsm_state", "N/A")
                fsm_reason_str = getattr(self, "fsm_reason", "")
                # tl_state_str = "RED" if tl_red else "green"
                # stop_state_str = "STOP sign" if stop_sign_present else "no sign"
                # tl_dist_str = f"{tl_distance:.1f}m" if tl_distance is not None else "N/A"
                # ss_dist_str = f"{stop_sign_distance:.1f}m" if stop_sign_distance is not None else "N/A"
                obstacle_ahead_fsm = bool(cargo.get("obstacle_ahead", False))


                txts = [
                    f"step: {self.step}",
                    f"cmd: {getattr(next_cmd, 'name', str(next_cmd))}",
                    f"speed: {v:.2f} m/s (target {plan_target_speed:.1f})",
                    f"steer: {control.steer:+.3f}",
                    f"throttle: {control.throttle:.3f}",
                    f"brake: {control.brake:.3f}",
                    f"obstacle_ahead: {obstacle_ahead_fsm}",
                    f"fsm_state: {fsm_state_str}",
                    f"fsm_reason: {fsm_reason_str}",
                    # f"tl_red: {tl_state_str} (dist {tl_dist_str})",
                    # f"stop_sign: {stop_state_str} (dist {ss_dist_str})",
                ]



                # Local coordinates of the next sparse global waypoint (for debugging)
                pos_xy_dbg = self._get_position_xy(data['gps_ll'])
                bearing_dbg = float(data['compass'])
                yaw_dbg = math.pi / 2.0 - bearing_dbg
                c_dbg, s_dbg = math.cos(yaw_dbg), math.sin(yaw_dbg)
                R_world_to_ego = np.array(
                    [[c_dbg,  s_dbg],
                     [-s_dbg, c_dbg]],
                    dtype=np.float32,
                )

                dn_next = float(next_wp[0] - pos_xy_dbg[0])
                de_next = float(next_wp[1] - pos_xy_dbg[1])
                x_fwd_next, y_left_next = (
                    R_world_to_ego @ np.array([de_next, dn_next], dtype=np.float32)
                ).tolist()

                # txts.append(
                #     f"next_wp_local: x_fwd={x_fwd_next:.1f}m, y_left={y_left_next:.1f}m"
                # )

                y0 = 28
                for i, t in enumerate(txts):
                    cv2.putText(
                        hud, t, (12, y0 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA,
                    )


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

                hud = cv2.resize(hud, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('rgb_front', hud)
                cv2.waitKey(1)

        except Exception:
            # If running headless, just ignore GUI errors
            print("[MyAutopilot][HUD] Exception in HUD block:")

        return control




    def destroy(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass



