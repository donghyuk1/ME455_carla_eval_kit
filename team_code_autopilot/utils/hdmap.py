#hdmap class for CARLA autonomous driving


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os
import math
import weakref
from typing import Optional, Tuple, List
import numpy as np
import carla
import cv2

import team_code_autopilot.utils.test_xodr as tx
import team_code_autopilot.utils.carla_vehicle_annotator as cva
import team_code_autopilot.utils.gemap_annotator as ga
from team_code_autopilot.utils.astar_planner import AStarPlanner  # ensure it's available

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




# -----------------------------------------------------------------------------
# HDMap class
# -----------------------------------------------------------------------------

class HDMap:

    def __init__(
        self,
        host: str,
        port: int,
        role: str,
        *,
        cam_res: Tuple[int, int] = (640, 400),
        frustum_max_dist: float = 61.0,
        sensor_tick: float = 0.0,
        is_visualize: bool = True,
    ):
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world: carla.World = self.client.get_world()
        self.map: carla.Map = self.world.get_map()

        self.role = role

        self.w, self.h = cam_res
        self.max_dist = float(frustum_max_dist)
        self.sensor_tick = float(sensor_tick)
        self.is_visualize = bool(is_visualize)
        # self.use_masked_points = False  # whether to filter hdmap points by frustum masks

        if self.is_visualize:
            cv2.namedWindow("HDMap_Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HDMap_Debug", 800, 800)

        # Ego and camera
        # self.is_ego_ready = False
        # self.ego: Optional[carla.Actor] = self._find_ego_by_role(self.world, self.role)
        # if(self.ego is not None):
        #     self.is_ego_ready = True
        self.ego: Optional[carla.Actor] = None
        self.camera_units = []
        self.sensors = []  # for cleanup



        # Parsed OpenDRIVE polylines (static)
        self.center_pts: Optional[np.ndarray] = None
        self.divider_pts: Optional[np.ndarray] = None
        self.bound_pts: Optional[np.ndarray] = None
        self.cross_pts: Optional[np.ndarray] = None
        self.idxes = None  # tuple of index arrays mapping each point to its polyline id

        self._prepare_hdmap_points()  # fill arrays once

        # Masks (dynamic, per tick)
        self.center_mask = np.zeros(len(self.center_pts), dtype=bool) if self.center_pts is not None else None
        self.divider_mask = np.zeros(len(self.divider_pts), dtype=bool) if self.divider_pts is not None else None
        self.bound_mask = np.zeros(len(self.bound_pts), dtype=bool) if self.bound_pts is not None else None
        self.cross_mask = np.zeros(len(self.cross_pts), dtype=bool) if self.cross_pts is not None else None

        self.center_mask_clustered = None
        self.divider_mask_clustered = None
        self.bound_mask_clustered = None
        self.cross_mask_clustered = None

        # Snapshots of actors (last tick)
        self._ego_snap = None
        self._vehicle_snaps = []
        self._walker_snaps = []
        self._vehicle_bboxes_3d = []     # list of bboxes for vehicles
        self._pedestrian_bboxes_3d = []  # list of bboxes for walkers

        # ---------- NEW: A* + route endpoints only ----------
        from team_code_autopilot.utils.astar_planner import AStarPlanner

        self._AStarPlannerClass = AStarPlanner  # just store the class
        self._astar_planner: Optional[AStarPlanner] = None
        self._astar_neighbor_radius: float = 4.0  # meters

        # endpoints in world frame (x, y, z)
        self._curr_wp_world: Optional[np.ndarray] = None
        self._next_wp_world: Optional[np.ndarray] = None

        # last endpoints used to build the path (for change detection)
        self._last_curr_wp_world: Optional[np.ndarray] = None
        self._last_next_wp_world: Optional[np.ndarray] = None

        # A* output: centerline path between curr_wp and next_wp
        self._local_center_path: Optional[np.ndarray] = None  # (M, 3)
        self._local_center_path_progress: int = 0
        self._current_lane_path: Optional[np.ndarray] = None  # (M, 3)
        self._right_lane_path: Optional[np.ndarray] = None  # (M, 3)
        self._left_lane_path: Optional[np.ndarray] = None  # (M, 3)

        # projection / validity flags
        self._projection_max_dist: float = 10.0  # m
        self._projection_ok: bool = False
        self._path_valid: bool = False
        self._current_path_valid: bool = False
        self._right_path_valid: bool = False
        self._left_path_valid: bool = False
 


    def __del__(self):
        try:
            cam = self._cam_ref() if self._cam_ref else None
            if cam and cam.is_alive:
                cam.destroy()
        except Exception:
            pass

        if self.is_visualize:
            try:
                cv2.destroyWindow("HDMap_Debug")
            except Exception:
                pass



    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _spawn_rgb(
        self,
        world: carla.World,
        parent: carla.Actor,
        image_w: int,
        image_h: int,
        sensor_tick: float,
        loc=(0.6, 0.0, 2.2),
        rot=(-8.0, 0.0, 0.0),
    ) -> Optional[carla.Sensor]:
        """Spawn a small RGB sensor for frustum filtering (no display)."""
        try:
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(image_w))
            bp.set_attribute("image_size_y", str(image_h))
            bp.set_attribute("sensor_tick", str(sensor_tick))
            t = carla.Transform(
                carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
            )
            return world.try_spawn_actor(bp, t, attach_to=parent)
        except Exception:
            return None


    def _prepare_hdmap_points(self):
        xodr = self.map.to_opendrive()
        self.center_pts, self.divider_pts, self.bound_pts, self.cross_pts, self.idxes = tx.extract_waypoints(xodr)
    


    def _reacquire_ego_and_cameras_if_needed(self) -> bool:
        """
        Find ego by role_name if needed, and spawn RGB cameras once ego exists.
        Returns True if ego & cameras are ready, False otherwise.
        """
        # 1) Find ego if missing or dead
        if self.ego is None or not self.ego.is_alive:
            self.ego = self._find_ego_by_role(self.world, self.role)
            if self.ego is None:
                self.is_ego_ready = False
                return False

        # 2) Spawn cameras once (when we first have a valid ego)
        if not self.camera_units:
            for cam_name, cam_cfg in CAMERA_SETUPS.items():
                if not cam_cfg.get('enabled', True):
                    continue
                if not cam_name.startswith('RGB'):
                    continue

                loc = cam_cfg.get('location', (0.5, 0.0, 2.2))
                rot = cam_cfg.get('rotation', (-8.0, 0.0, 0.0))

                sensor = self._spawn_rgb(
                    self.world,
                    self.ego,
                    self.w,
                    self.h,
                    sensor_tick=self.sensor_tick,
                    loc=loc,
                    rot=rot,
                )
                if sensor is None:
                    print(f"[HDMap] Failed to spawn {cam_name}")
                    continue

                self.sensors.append(sensor)
                self.camera_units.append({
                    'name': cam_name,
                    'sensor': sensor,
                    'depth_sensor': None,
                    'dirs': {},
                })

        self.is_ego_ready = True
        return True


    @staticmethod
    def _find_ego_by_role(world: carla.World, role_name: str) -> Optional[carla.Actor]:
        actors = world.get_actors().filter("vehicle.*")
        for a in actors:
            if a.attributes.get("role_name", "") == role_name:
                return a
        return None
    

    @staticmethod
    def _dist2(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))


    def _group_points_by_id(
        self,
        pts: Optional[np.ndarray],
        idx_array: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ):
        """
        Group points into polylines using idx_array (one ID per point).

        Returns:
            - list of (Mi, 3) np.ndarrays, each corresponding to a single polyline
            - or a single (N, 3) np.ndarray if idx_array is None (fallback)
        """
        if pts is None:
            return None

        pts = np.asarray(pts)
        if pts.size == 0:
            return pts  # empty array

        # If we don't have index information, just apply mask and return flat points
        if idx_array is None:
            if mask is not None:
                return pts[mask]
            return pts

        idx_array = np.asarray(idx_array)
        assert len(idx_array) == len(pts), "idx_array length must match pts length"

        if mask is not None:
            pts = pts[mask]
            idx_array = idx_array[mask]

        if len(pts) == 0:
            return []

        poly_ids = np.unique(idx_array)
        grouped = [pts[idx_array == pid] for pid in poly_ids]
        return grouped


    def _find_closest_center_cluster(
        self,
        location_xy: np.ndarray,
    ):
        """
        Given a 2D world location (x, y), find the closest centerline cluster.

        Uses self.center_mask_clustered, which is a list of (Mi, 3) arrays
        returned by _group_points_by_id(...).

        Args:
            location_xy: np.ndarray shape (2,) or (3,), world-frame (x, y [, z])

        Returns:
            (best_idx, best_cluster_pts, best_dist)

            best_idx:
                - index into self.center_mask_clustered
                - -1 if not found / no clusters
            best_cluster_pts:
                - (Mi, 3) np.ndarray of the chosen cluster
                - None if no cluster found
            best_dist:
                - minimal Euclidean distance [m] from location_xy to any point in that cluster
                - float("inf") if no cluster found
        """
        # Normalize input location to [x, y]
        loc = np.asarray(location_xy, dtype=float).reshape(-1)
        if loc.size >= 2:
            loc_xy = loc[:2]
        else:
            raise ValueError("location_xy must have at least 2 elements (x, y).")

        # Make sure clusters exist
        clusters = getattr(self, "center_mask_clustered", None)
        if clusters is None or len(clusters) == 0:
            print("[HDMap] Warning: center_mask_clustered is empty; cannot find closest cluster.")
            return -1, None, float("inf")

        best_idx = -1
        best_d2 = float("inf")
        best_cluster = None

        for i, seg in enumerate(clusters):
            if seg is None or len(seg) == 0:
                continue

            seg = np.asarray(seg)
            # seg is (Mi, 3) -> take XY
            seg_xy = seg[:, :2]

            diff = seg_xy - loc_xy[None, :]
            d2 = np.einsum("ij,ij->i", diff, diff)
            local_min = float(d2.min())

            if local_min < best_d2:
                best_d2 = local_min
                best_idx = i
                best_cluster = seg

        if best_idx < 0:
            return -1, None, float("inf")

        return best_idx, best_cluster, math.sqrt(best_d2)


    def _find_neighbor_lanes(
        self,
        curr_cluster_idx: int,
        curr_cluster_pts: np.ndarray,
        ref_point_xy: np.ndarray,
        lane_offset_min: float = 2.0,
        lane_offset_max: float = 5.0,
        parallel_cos_min: float = 0.866,  # ~ cos(30 deg)
    ):
        """
        Given the current lane cluster and a reference point (x, y) on it,
        find the closest lane to the LEFT and RIGHT.

        A lane is considered "adjacent" if:
          - its centerline is roughly parallel to the current lane
          - its lateral distance is within [lane_offset_min, lane_offset_max].

        Returns:
            (left_cluster_pts or None, right_cluster_pts or None)
        """
        clusters = getattr(self, "center_mask_clustered", None)
        if clusters is None or len(clusters) == 0:
            return None, None

        curr_pts = np.asarray(curr_cluster_pts, dtype=float)
        if curr_pts.ndim != 2 or curr_pts.shape[0] < 2:
            return None, None

        curr_xy = curr_pts[:, :2]

        # --- 1) reference point on current lane ---
        ref_xy = np.asarray(ref_point_xy, dtype=float).reshape(-1)[:2]
        diff = curr_xy - ref_xy[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        j = int(np.argmin(d2))  # index on current lane closest to ref_xy

        # tangent on current lane at that index
        if j < curr_xy.shape[0] - 1:
            t = curr_xy[j + 1] - curr_xy[j]
        elif j > 0:
            t = curr_xy[j] - curr_xy[j - 1]
        else:
            t = np.array([1.0, 0.0], dtype=float)

        if np.linalg.norm(t) < 1e-6:
            t = np.array([1.0, 0.0], dtype=float)
        t = t / (np.linalg.norm(t) + 1e-9)  # unit tangent

        ref_on_lane = curr_xy[j]

        best_left = None
        best_right = None
        best_left_dist = float("inf")
        best_right_dist = float("inf")

        # --- 2) scan all other lane clusters ---
        for k, seg in enumerate(clusters):
            if k == curr_cluster_idx or seg is None or len(seg) < 2:
                continue

            seg = np.asarray(seg, dtype=float)
            seg_xy = seg[:, :2]

            # closest point on candidate lane to ref_on_lane
            diff_seg = seg_xy - ref_on_lane[None, :]
            d2_seg = np.einsum("ij,ij->i", diff_seg, diff_seg)
            min_idx = int(np.argmin(d2_seg))
            other_point = seg_xy[min_idx]
            offset = other_point - ref_on_lane
            dist = float(np.linalg.norm(offset))

            # reject if too close (same lane) or too far (not adjacent)
            if dist < 1e-3:
                continue
            if dist < lane_offset_min or dist > lane_offset_max:
                continue

            # tangent of candidate lane near that point
            if min_idx < seg_xy.shape[0] - 1:
                t_other = seg_xy[min_idx + 1] - seg_xy[min_idx]
            elif min_idx > 0:
                t_other = seg_xy[min_idx] - seg_xy[min_idx - 1]
            else:
                t_other = np.array([1.0, 0.0], dtype=float)

            if np.linalg.norm(t_other) < 1e-6:
                continue
            t_other = t_other / (np.linalg.norm(t_other) + 1e-9)

            # require roughly parallel centerlines
            parallel = abs(float(np.dot(t, t_other)))
            if parallel < parallel_cos_min:
                continue

            # sign of cross(t, offset) tells left (>0) vs right (<0)
            cross_z = t[0] * offset[1] - t[1] * offset[0]
            if abs(cross_z) < 1e-3:
                # essentially collinear / same lane direction, no clear lateral side
                continue

            if cross_z > 0:  # candidate is on the LEFT
                if dist < best_left_dist:
                    best_left_dist = dist
                    best_left = seg
            else:  # candidate is on the RIGHT
                if dist < best_right_dist:
                    best_right_dist = dist
                    best_right = seg

        # If no lane satisfied the constraints, we return None on that side.
        return best_left, best_right


    def _closest_global_idx_ahead(self, location_xy: np.ndarray, min_forward_dot: float = 0.0) -> int:
        """
        Very simple "closest ahead" selector on the precomputed global waypoints.

        min_forward_dot: if > 0, will prefer points broadly aligned with ego heading
        """
        if len(self.global_waypoints) == 0:
            return 0

        # Closest by distance first
        xy = self.global_waypoints[:, :2]
        diffs = xy - location_xy[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(d2))

        # Optional heading filtering (requires ego transform)
        if self._ego_snap is not None and min_forward_dot > 0.0:
            ego_yaw = np.radians(self._ego_snap["rotation"]["yaw"])
            f = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
            # walk forward until dot >= threshold (guard against looping)
            k = idx
            for _ in range(min(50, len(xy))):
                dir_vec = xy[k] - location_xy
                if np.linalg.norm(dir_vec) < 1e-6:
                    break
                dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                if float(np.dot(dir_vec, f)) >= min_forward_dot:
                    idx = k
                    break
                k = (k + 1) % len(xy)

        return idx


    def _update_hdmap_info(self):
        """
        Refresh dynamic info:
          - ego & other actors snapshot
          - camera-based frustum masks for center/divider/bound/cross
        """
        
        # 1) Keep ego & cameras valid; if not ready, skip this tick gracefully
        if not self._reacquire_ego_and_cameras_if_needed():
            return


        # 2) Sync to the world tick (non-blocking-ish)
        # snapshot = self.world.wait_for_tick(2.0)
        snapshot = self.world.get_snapshot()
        if snapshot is None:
            return

        # 3) Build ego/others snapshots for geometry/bbox utils
        try:
            ego_snap = cva.snap_processing([self.ego], snapshot)[0]
        except Exception:
            return

        actors = self.world.get_actors()
        vehicles = [a for a in actors.filter("vehicle.*") if a.id != self.ego.id]
        walkers = list(actors.filter("walker.*"))

        vehicle_snaps = cva.snap_processing(vehicles, snapshot) if vehicles else []
        walker_snaps = cva.snap_processing(walkers, snapshot) if walkers else []

        self._ego_snap = ego_snap
        self._vehicle_snaps = vehicle_snaps
        self._walker_snaps = walker_snaps

        # Compute 3D bounding boxes for vehicles and pedestrians
        try:
            if vehicle_snaps:
                self._vehicle_bboxes_3d = ga.get_vehicle_bbox(
                    ego_snap,
                    vehicle_snaps,
                    radius=self.max_dist,
                )
            else:
                self._vehicle_bboxes_3d = []

            if walker_snaps:
                self._pedestrian_bboxes_3d = ga.get_vehicle_bbox(
                    ego_snap,
                    walker_snaps,
                    radius=self.max_dist,
                )
            else:
                self._pedestrian_bboxes_3d = []
        except Exception as e:
            # Fail gracefully; we don't want bbox errors to break HDMap
            print(f"[HDMap] Warning: get_vehicle_bbox failed: {e}")
            self._vehicle_bboxes_3d = []
            self._pedestrian_bboxes_3d = []


        # 4) Update masks via frustum
        if any(x is None for x in (self.center_pts, self.divider_pts, self.bound_pts, self.cross_pts, self.idxes)):
            # Should not happen (we parse once in __init__), but guard anyway
            self._prepare_hdmap_points()

        center_mask = np.zeros(len(self.center_pts), dtype=bool)
        divider_mask = np.zeros(len(self.divider_pts), dtype=bool)
        bound_mask = np.zeros(len(self.bound_pts), dtype=bool)
        cross_mask = np.zeros(len(self.cross_pts), dtype=bool)


        for unit in self.camera_units:
            if not unit['name'].startswith('RGB'):
                continue
            cam_actor = unit['sensor']
            if cam_actor is None or not cam_actor.is_alive:
                continue
            center_mask, divider_mask, bound_mask, cross_mask = ga.filter_by_cameras(
                camera=cam_actor,
                ego_vehicle=ego_snap,
                center_pts=self.center_pts,
                divider_pts=self.divider_pts,
                bound_pts=self.bound_pts,
                cross_pts=self.cross_pts,
                masks=(center_mask, divider_mask, bound_mask, cross_mask),
                max_dist=self.max_dist,
                min_dist=0.1
            )

        self.center_mask = center_mask
        self.divider_mask = divider_mask
        self.bound_mask = bound_mask
        self.cross_mask = cross_mask


    def _update_waypoint_cursor(self):
        """Maintain a simple forward-moving cursor along global sparse waypoints."""
        if self._ego_snap is None or len(self.global_waypoints) == 0:
            return

        ego_loc = np.array([self._ego_snap["location"]["x"], self._ego_snap["location"]["y"]], dtype=float)
        idx = self._closest_global_idx_ahead(ego_loc, min_forward_dot=0.1)

        # Advance cursor toward idx but never go backward (assumes roughly forward progress)
        if idx >= self._wp_idx:
            self._wp_idx = idx
        else:
            # if wrap-around or respawn, allow reset if it's "much" earlier
            if (self._wp_idx - idx) > 20:
                self._wp_idx = idx


    def _ensure_astar_planner(self):
        """Lazily create A* planner over center_pts XY."""
        if self._astar_planner is not None:
            return

        if self.center_pts is None or len(self.center_pts) == 0:
            self._prepare_hdmap_points()

        if self.center_pts is None or len(self.center_pts) == 0:
            return

        # center_xy = self.center_pts[:, :2].astype(np.float32)
        center_xy = self.center_pts[:, :2].astype(np.float32) 
        self._astar_planner = self._AStarPlannerClass(
            points=center_xy,
            neighbor_radius=self._astar_neighbor_radius,
            heuristic_weight=2.0,
        )


    def _project_to_center_idx(self, wp_xy: np.ndarray) -> Tuple[int, float]:
        """
        Project a world waypoint (x,y) to the nearest center_pts node.

        Returns:
            (idx, dist) where idx is index into self.center_pts, dist in meters.
            If center_pts is empty, returns (-1, inf).
        """
        if self.center_pts is None or len(self.center_pts) == 0:
            print("[HDMap] Warning: center_pts is empty; cannot project waypoint.")
            return -1, float("inf")

        center_xy = self.center_pts[:, :2]  # (N, 2)
        diff = center_xy - wp_xy[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        k = int(np.argmin(d2))
        d = float(np.sqrt(d2[k]))
        return k, d
    

    def _make_hdmap_image_ego(
        self,
        center_pts: np.ndarray,
        divider_pts: np.ndarray,
        bound_pts: np.ndarray,
        cross_pts: np.ndarray,
        ego_snap,
        img_size=(800, 800),
        pixels_per_meter: float = 4.0,
        curr_wp_world: Optional[np.ndarray] = None, # adding waypoint features
        next_wp_world: Optional[np.ndarray] = None, # adding waypoint features
        path_pts: Optional[np.ndarray] = None,
        vehicle_bboxes_3d: Optional[list] = None,
        pedestrian_bboxes_3d: Optional[list] = None,
        current_lane_pts: Optional[np.ndarray] = None,
        left_lane_pts: Optional[np.ndarray] = None,
        right_lane_pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create an ego-centered HD map image, similar to simple_hdmap_image()
        in hdmap_vis.py.

        - center_pts, divider_pts, bound_pts, cross_pts: (N,3) arrays in world frame.
        - ego_snap: cva.snap_processing(...) result for ego
        - img_size: output image size (h, w)
        - pixels_per_meter: scaling of meters -> pixels
        """
        h, w = img_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Ego pose from snap
        ego_tf = ego_snap.get_transform()
        ego_loc = ego_tf.location
        ego_rot = ego_tf.rotation

        ego_x = ego_loc.x
        ego_y = ego_loc.y
        ego_yaw = math.radians(ego_rot.yaw)

        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)

        # Image center == ego position
        cx = w // 2
        cy = h // 2

        def world_to_ego_pixel(x, y):
            # 1) world -> ego frame (Y_body forward, X_body left)
            dx = x - ego_x
            dy = y - ego_y

            Y_body =  cos_yaw * dx + sin_yaw * dy   # forward(+)
            X_body = -sin_yaw * dx + cos_yaw * dy   # left(+)

            # 2) ego -> pixel
            ix = int(round(cx + X_body * pixels_per_meter))
            iy = int(round(cy - Y_body * pixels_per_meter))  # forward => up
            return ix, iy

        # def draw_points(pts, color, radius=1):
        #     if pts is None or len(pts) == 0:
        #         return
        #     pts = np.asarray(pts)
        #     xs = pts[:, 0]
        #     ys = pts[:, 1]
        #     for x, y in zip(xs, ys):
        #         ix, iy = world_to_ego_pixel(x, y)
        #         if 0 <= ix < w and 0 <= iy < h:
        #             cv2.circle(img, (ix, iy), radius, color, -1)

        def draw_points(pts, color, radius=1):
            """
            Draw points for:
              - a single (N, 3) numpy array, or
              - a list / tuple of such arrays (polylines).
            """
            if pts is None:
                return

            # If we got polyline groups, recurse on each
            if isinstance(pts, (list, tuple)):
                for seg in pts:
                    draw_points(seg, color, radius)
                return

            pts = np.asarray(pts)
            if pts.size == 0:
                return

            xs = pts[:, 0]
            ys = pts[:, 1]
            for x, y in zip(xs, ys):
                ix, iy = world_to_ego_pixel(x, y)
                if 0 <= ix < w and 0 <= iy < h:
                    cv2.circle(img, (ix, iy), radius, color, -1)


        def draw_polyline(pts, color, thickness=2):
            if pts is None:
                return
            pts = np.asarray(pts)
            if len(pts) < 2:
                return
            xs = pts[:, 0]
            ys = pts[:, 1]
            prev_px = None
            for x, y in zip(xs, ys):
                ix, iy = world_to_ego_pixel(x, y)
                if 0 <= ix < w and 0 <= iy < h:
                    if prev_px is not None:
                        cv2.line(img, prev_px, (ix, iy), color, thickness)
                    prev_px = (ix, iy)
                else:
                    prev_px = None

        def draw_bbox_centers(bboxes, color, half_size_px=4):
            """
            Draw simple square bboxes using the center of each 3D bbox.

            Supports:
              - dict entries from ga.get_vehicle_bbox (with 'corners_world' / 'gt_array')
              - arrays/lists of corners: (N, 3)
              - single 3D point: (3,)
            """
            if bboxes is None:
                return

            for bb in bboxes:
                if bb is None:
                    continue

                center_world = None

                # ----- Case 1: dict from ga.get_vehicle_bbox -----
                if isinstance(bb, dict):
                    # Prefer corners_world if available
                    if "corners_world" in bb and bb["corners_world"]:
                        corners = np.asarray(bb["corners_world"], dtype=float)
                        if corners.ndim == 2 and corners.shape[1] >= 2:
                            center_world = corners.mean(axis=0)  # (3,)
                    # Fallback: gt_array (first 3 = x,y,z)
                    if center_world is None and "gt_array" in bb:
                        arr = np.asarray(bb["gt_array"], dtype=float)
                        if arr.size >= 3:
                            center_world = arr[:3]

                # ----- Case 2: not a dict -> interpret as array-like -----
                else:
                    bb_arr = np.asarray(bb, dtype=float)

                    if bb_arr.size == 0:
                        continue

                    if bb_arr.ndim == 0:
                        # scalar, no spatial info
                        continue
                    elif bb_arr.ndim == 1:
                        # single 3D point
                        if bb_arr.shape[0] >= 2:
                            center_world = bb_arr
                    else:
                        # corners (N, 3)
                        if bb_arr.shape[1] >= 2:
                            center_world = bb_arr.mean(axis=0)

                # If we still couldn't get a center, skip this bbox
                if center_world is None:
                    continue

                # Need at least x, y
                if len(center_world) < 2:
                    continue

                ix, iy = world_to_ego_pixel(center_world[0], center_world[1])

                if 0 <= ix < w and 0 <= iy < h:
                    x1 = int(ix - half_size_px)
                    y1 = int(iy - half_size_px)
                    x2 = int(ix + half_size_px)
                    y2 = int(iy + half_size_px)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        # Colors
        color_center  = (0, 255, 255)  # centerline
        color_divider = (0, 255, 0)    # lane divider
        color_bound   = (255, 255, 0)  # road boundary
        color_cross   = (255, 0, 255)  # crosswalk

        # Draw layers
        draw_points(center_pts,  color_center,  radius=1)
        draw_points(divider_pts, color_divider, radius=1)
        draw_points(bound_pts,   color_bound,   radius=1)
        draw_points(cross_pts,   color_cross,   radius=1)

        # Vehicles: blue squares
        draw_bbox_centers(vehicle_bboxes_3d, color=(255, 0, 0), half_size_px=5)
        # Pedestrians: magenta squares
        draw_bbox_centers(pedestrian_bboxes_3d, color=(255, 0, 255), half_size_px=4)

        # Ego marker at center
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(img, "EGO", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ----- Optional: draw current & next global waypoints -----
        # curr_wp_world in RED
        if curr_wp_world is not None:
            curr_wp_world = np.asarray(curr_wp_world, dtype=float).reshape(-1)
            ix, iy = world_to_ego_pixel(curr_wp_world[0], curr_wp_world[1])
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(img, (ix, iy), 6, (0, 0, 255), -1)
                cv2.putText(img, "C", (ix + 4, iy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # next_wp_world in GREEN
        if next_wp_world is not None:
            next_wp_world = np.asarray(next_wp_world, dtype=float).reshape(-1)
            ix, iy = world_to_ego_pixel(next_wp_world[0], next_wp_world[1])
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(img, (ix, iy), 6, (0, 255, 0), -1)
                cv2.putText(img, "N", (ix + 4, iy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # if path_pts is not None and len(path_pts) > 1:
        #     # path_pts is expected to be (M, 3) in world coords (self._local_center_path)
        #     draw_polyline(path_pts, color=(255, 0, 0), thickness=2)
        # else:
        #     print("No path_pts to draw in HD map image.")

        # ----- Draw lane centerlines (current / left / right) -----
        # Use thin lines so they don't dominate the HD map
        if current_lane_pts is not None and len(current_lane_pts) > 1:
            # white for current lane
            draw_polyline(current_lane_pts, color=(255, 255, 255), thickness=1)

        if left_lane_pts is not None and len(left_lane_pts) > 1:
            # cyan-ish for left lane
            draw_polyline(left_lane_pts, color=(255, 255, 0), thickness=1)

        if right_lane_pts is not None and len(right_lane_pts) > 1:
            # yellow-ish for right lane
            draw_polyline(right_lane_pts, color=(0, 255, 255), thickness=1)


        if path_pts is not None and len(path_pts) > 1:
            # Check if route overlaps any vehicle bbox (ground footprint)
            route_blocked = False
            if vehicle_bboxes_3d:
                for bb in vehicle_bboxes_3d:
                    if bb is None:
                        continue
                    if self._route_overlaps_bbox(path_pts, bb, margin=0.5):
                        route_blocked = True
                        break

            # If not overlapped → BLUE (as before), if overlapped → RED
            color_free = (255, 0, 0)   # blue in BGR
            color_blocked = (0, 0, 255)  # red in BGR
            route_color = color_blocked if route_blocked else color_free

            # path_pts is expected to be (M, 3) in world coords (self._local_center_path)
            draw_polyline(path_pts, color=route_color, thickness=2)
        else:
            print("No path_pts to draw in HD map image.")

        return img


    def _draw_hdmap_debug(self):
        """
        Visualize current HD-map in an ego-centered frame:

          - visible center/divider/bound/cross (using masks)
          - ego vehicle at image center

        For now: no A* route, no global waypoint visualization.
        """
        if not self.is_visualize:
            return
        if self._ego_snap is None:
            return

        center_idx, divider_idx, bound_idx, cross_idx = self.idxes

        self.center_mask_clustered  = self._group_points_by_id(self.center_pts,  center_idx,  self.center_mask)
        self.divider_mask_clustered = self._group_points_by_id(self.divider_pts, divider_idx, self.divider_mask)
        self.bound_mask_clustered   = self._group_points_by_id(self.bound_pts,   bound_idx,   self.bound_mask)
        self.cross_mask_clustered   = self._group_points_by_id(self.cross_pts,   cross_idx,   self.cross_mask)


        path_pts = self._local_center_path if self.has_valid_local_path else None

        current_lane_pts = self._current_lane_path if self._current_path_valid else None
        left_lane_pts = self._left_lane_path if self._left_path_valid else None
        right_lane_pts = self._right_lane_path if self._right_path_valid else None


        # Delegate image creation to modular helper (same as hdmap_vis.py)
        img = self._make_hdmap_image_ego(
            center_pts=self.center_mask_clustered,
            divider_pts=self.divider_mask_clustered,
            bound_pts=self.bound_mask_clustered,
            cross_pts=self.cross_mask_clustered,
            ego_snap=self._ego_snap,
            img_size=(800, 800),
            pixels_per_meter=4.0,
            curr_wp_world=self._curr_wp_world,
            next_wp_world=self._next_wp_world,
            path_pts=path_pts,
            vehicle_bboxes_3d=self._vehicle_bboxes_3d,
            pedestrian_bboxes_3d=self._pedestrian_bboxes_3d,
            current_lane_pts=current_lane_pts,
            left_lane_pts=left_lane_pts,
            right_lane_pts=right_lane_pts,
        )

        cv2.imshow("HDMap_Debug", img)
        cv2.waitKey(1)


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


    @property
    def has_valid_local_path(self) -> bool:
        """
        True if last A* succeeded AND projection was OK AND we have non-empty path.
        """
        return (
            self._path_valid
            and self._local_center_path is not None
            and len(self._local_center_path) > 0
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def tick(self):
        """Call this every control cycle to update masks and actor states."""
        self._update_hdmap_info()
        # self._update_waypoint_cursor()

        # NEW: visualize current map + waypoints + route
        if self.is_visualize:
            self._draw_hdmap_debug()


    def update_route_between_globals(
        self,
        curr_wp_world: Optional[np.ndarray],
        next_wp_world: Optional[np.ndarray],
    ) -> None:
        """
        Update the A* path between two global waypoints in WORLD coordinates.

        Arguments:
          curr_wp_world : np.array([x,y,z]) or None
          next_wp_world : np.array([x,y,z]) or None

        Behavior:
          - If next_wp_world is None -> do NOT update; keep existing path.
          - If (curr_wp_world, next_wp_world) is unchanged (within tolerance),
            do NOT recompute A*.
          - Otherwise:
              * project both to nearest center_pts (XY)
              * if either is farther than projection_max_dist -> mark failure
              * else run A* between those two center_pts nodes and store path.
        """
        # 1) no next waypoint => nothing to update
        if next_wp_world is None:
            return

        if curr_wp_world is None:
            # can't define a segment if we don't know the current wp
            self._path_valid = False
            self._projection_ok = False
            self._local_center_path = None
            self._local_center_path_progress = 0
            return

        # # Normalize to np.array shape (3,)
        curr_wp_world = np.asarray(curr_wp_world, dtype=float).reshape(-1)
        next_wp_world = np.asarray(next_wp_world, dtype=float).reshape(-1)

        # 2) if pair unchanged (within small tolerance), skip recomputing
        if (
            self._last_curr_wp_world is not None
            and self._last_next_wp_world is not None
            and np.allclose(curr_wp_world, self._last_curr_wp_world, atol=1e-3)
            and np.allclose(next_wp_world, self._last_next_wp_world, atol=1e-3)
            and self.has_valid_local_path
        ):
            # nothing changed; keep existing path
            return

        # # Remember this pair
        self._curr_wp_world = curr_wp_world
        self._next_wp_world = next_wp_world
        self._last_curr_wp_world = curr_wp_world.copy()
        self._last_next_wp_world = next_wp_world.copy()
        self._current_path_valid = False
        self._right_path_valid = False
        self._left_path_valid = False

        # wp_xy is a (2,) numpy array in world coords
        idx, cluster_pts, dist = self._find_closest_center_cluster(self._next_wp_world)

        if idx >= 0:
            # cluster_pts is (Mi, 3) – this will be your “local route” to follow
            self._local_center_path = cluster_pts.copy()
            self._local_center_path_progress = 0
            self._path_valid = True
        else:
            self._path_valid = False

        # Check projection distance for current lane
        self._projection_ok = (dist_curr <= self._projection_max_dist)

        # Current lane path == chosen cluster
        self._current_lane_path = curr_cluster_pts.copy()
        self._current_path_valid = self._projection_ok

        # For backward compatibility, use current lane as "local path"
        self._local_center_path = curr_cluster_pts.copy()
        self._local_center_path_progress = 0
        self._path_valid = self._projection_ok

        # 4) Estimate left / right neighbor lanes around curr_wp_world
        left_seg, right_seg = self._find_neighbor_lanes(
            curr_cluster_idx=curr_idx,
            curr_cluster_pts=curr_cluster_pts,
            ref_point_xy=curr_wp_world[:2],
        )

        if left_seg is not None and len(left_seg) > 0:
            self._left_lane_path = np.asarray(left_seg, dtype=float).copy()
            self._left_path_valid = True
        else:
            self._left_lane_path = None
            self._left_path_valid = False

        if right_seg is not None and len(right_seg) > 0:
            self._right_lane_path = np.asarray(right_seg, dtype=float).copy()
            self._right_path_valid = True
        else:
            self._right_lane_path = None
            self._right_path_valid = False


    def get_next_waypoint(self, location: Optional[Tuple[float, float, float]] = None, lookahead: int = 10) -> np.ndarray:
        """
        Returns a single next waypoint from the global sparse path.

        Args:
          location : optional (x,y[,z]) to override ego location for selection
          lookahead: how far ahead of the cursor to return (clamped)

        Returns:
          waypoint as np.array([x,y,z])
        """
        if len(self.global_waypoints) == 0:
            return np.zeros(3)

        if location is not None:
            loc = np.array(location[:2], dtype=float)
            idx = self._closest_global_idx_ahead(loc, min_forward_dot=0.0)
        else:
            idx = self._wp_idx

        k = int(min(idx + max(1, lookahead), len(self.global_waypoints) - 1))
        return self.global_waypoints[k]

    
    def get_nextlane_waypoint(self, location: Optional[Tuple[float, float, float]] = None, radius: float = 50.0) -> np.ndarray:
        """
        Returns a mid-lane waypoint near/ahead of the ego using visible centerlines.
        Simple heuristic: choose nearest visible centerline point ahead of ego.

        Args:
          location: optional (x,y[,z]) to override ego location
          radius  : search radius (m)

        Returns:
          waypoint as np.array([x,y,z]); falls back to global next if none found
        """
        if self.center_pts is None or self.center_mask is None:
            return self.get_next_waypoint(location)

        if location is not None:
            loc3 = np.array(location, dtype=float)
        elif self._ego_snap is not None:
            loc3 = np.array([
                self._ego_snap["location"]["x"],
                self._ego_snap["location"]["y"],
                self._ego_snap["location"]["z"],
            ], dtype=float)
        else:
            return self.get_next_waypoint(location)

        visible = self.center_pts[self.center_mask]
        if len(visible) == 0:
            return self.get_next_waypoint(location)

        # Filter by radius
        d2 = np.sum((visible - loc3[None, :])**2, axis=1)
        keep = d2 <= (radius * radius)
        if not np.any(keep):
            # too far; just take closest visible
            k = int(np.argmin(d2))
            return visible[k]

        vis_near = visible[keep]
        d2n = np.sum((vis_near - loc3[None, :])**2, axis=1)

        # Prefer points "ahead" using ego yaw if available
        if self._ego_snap is not None:
            ego_yaw = np.radians(self._ego_snap["rotation"]["yaw"])
            f = np.array([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=float)
            ahead_mask = []
            for p in vis_near:
                dir_xy = p[:2] - loc3[:2]
                if np.linalg.norm(dir_xy) < 1e-6:
                    ahead_mask.append(False)
                else:
                    ahead_mask.append(np.dot(dir_xy / (np.linalg.norm(dir_xy) + 1e-9), f) > 0)
            ahead_mask = np.array(ahead_mask, dtype=bool)
            if np.any(ahead_mask):
                vis_near = vis_near[ahead_mask]
                d2n = np.sum((vis_near - loc3[None, :])**2, axis=1)

        k = int(np.argmin(d2n))
        return vis_near[k]
    
    
    def is_obstacle_in_front(self, distance: float = 10.0, fov_deg: float = 30.0) -> bool:
        """
        Naive obstacle check: any vehicle/walker within a forward cone?
        """
        if self._ego_snap is None:
            return False

        ex, ey = self._ego_snap["location"]["x"], self._ego_snap["location"]["y"]
        ego_yaw = np.radians(self._ego_snap["rotation"]["yaw"])
        f = np.array([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=float)

        half_cos = np.cos(np.radians(fov_deg) / 2.0)
        dmax2 = distance * distance

        def _in_front(px, py):
            v = np.array([px - ex, py - ey], dtype=float)
            d2 = float(np.dot(v, v))
            if d2 > dmax2 or d2 < 1e-6:
                return False
            v /= (np.sqrt(d2) + 1e-9)
            return float(np.dot(v, f)) >= half_cos

        # vehicles
        for vs in self._vehicle_snaps:
            loc = vs["location"]
            if _in_front(loc["x"], loc["y"]):
                return True

        # walkers
        for ws in self._walker_snaps:
            loc = ws["location"]
            if _in_front(loc["x"], loc["y"]):
                return True

        return False


    def is_nextlane_free(self, distance: float = 20.0, clearance: float = 3.0) -> bool:
        # Check if the next-lane waypoint is free of obstacles within given distance.
        # Simple circular clearance check.
        pass

    
    def is_traffic_light_red(self):
        # Check if the traffic light in front of the ego vehicle is red
        pass





