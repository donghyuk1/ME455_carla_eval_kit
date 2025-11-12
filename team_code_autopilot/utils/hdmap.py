#hdmap class for CARLA autonomous driving


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# TODO : fill in the imports


import os
import weakref
from typing import Optional, Tuple, List
import numpy as np
import carla

import team_code_autopilot.utils.test_xodr as tx
import team_code_autopilot.utils.carla_vehicle_annotator as cva
import team_code_autopilot.utils.gemap_annotator as ga



# -----------------------------------------------------------------------------
# HDMap class
# -----------------------------------------------------------------------------

class HDMap:

    def __init__(
        self,
        host: str,
        port: int,
        role: str,
        global_waypoints: np.ndarray,
        *,
        cam_res: Tuple[int, int] = (640, 400),
        frustum_max_dist: float = 61.0,
        sensor_tick: float = 0.0,
    ):
        """
        Args:
          host/port       : CARLA server
          role            : role_name to identify ego (e.g. "hero")
          global_waypoints: (N,2) or (N,3) array of sparse path points (x,y[,z])
          cam_res         : RGB sensor resolution used for frustum filtering
          frustum_max_dist: max mask distance (meters)
          sensor_tick     : camera update tick (0.0 = every world tick)
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world: carla.World = self.client.get_world()
        self.map: carla.Map = self.world.get_map()

        self.role = role
        self.global_waypoints = np.asarray(global_waypoints, dtype=float)
        if self.global_waypoints.shape[1] == 2:
            # Add z=0 if not supplied
            self.global_waypoints = np.hstack([self.global_waypoints, np.zeros((len(self.global_waypoints), 1))])

        self.w, self.h = cam_res
        self.max_dist = float(frustum_max_dist)
        self.sensor_tick = float(sensor_tick)

        # Ego and camera
        self.ego: Optional[carla.Actor] = self._find_ego_by_role(self.world, self.role)
        self._cam_ref: Optional[weakref.ReferenceType] = None
        if self.ego:
            cam = self._spawn_rgb(self.world, self.ego, self.w, self.h, self.sensor_tick)
            if cam:
                self._cam_ref = weakref.ref(cam)

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

        # Snapshots of actors (last tick)
        self._ego_snap = None
        self._vehicle_snaps = []
        self._walker_snaps = []

        # Waypoint cursor (for simple “next” logic)
        self._wp_idx = 0

    def __del__(self):
        # Best-effort sensor cleanup
        try:
            cam = self._cam_ref() if self._cam_ref else None
            if cam and cam.is_alive:
                cam.destroy()
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


    def _update_hdmap_info(self):
        # Update internal map informations
        # Update the HDmap features : self.center_pts, self.divider_pts, self.bound_pts, self.cross_pts, self.idxes
        # Update the HDmap masks : 
     
        center_pts, divider_pts, bound_pts, cross_pts, idxes = self._prepare_hdmap_points(self.world)

        center_mask, divider_mask, bound_mask, cross_mask = ga.filter_by_cameras(
            camera=camera_actor,
            ego_vehicle=ego_snap,
            center_pts=center_pts,
            divider_pts=divider_pts,
            bound_pts=bound_pts,
            cross_pts=cross_pts,
            masks=(center_mask, divider_mask, bound_mask, cross_mask),
            max_dist=args.dist,
            min_dist=0.1
        )

        pass
        

    def _prepare_hdmap_points(self):
        """Parse the map's OpenDRIVE once and cache polylines + indices."""
        xodr = self.map.to_opendrive()
        center_pts, divider_pts, bound_pts, cross_pts, idxes = tx.extract_waypoints(xodr)
        # Ensure np arrays (N,3) float
        def _as3(a):
            a = np.asarray(a, dtype=float)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[1] == 2:
                a = np.hstack([a, np.zeros((len(a), 1))])
            return a

        self.center_pts = _as3(center_pts)
        self.divider_pts = _as3(divider_pts)
        self.bound_pts = _as3(bound_pts)
        self.cross_pts = _as3(cross_pts)
        self.idxes = idxes  # expected tuple/list of index arrays


    @staticmethod
    def _find_ego_by_role(world: carla.World, role_name: str) -> Optional[carla.Actor]:
        actors = world.get_actors().filter("vehicle.*")
        for a in actors:
            if a.attributes.get("role_name", "") == role_name:
                return a
        return None
    

    def _reacquire_ego_and_camera_if_needed(self):
        """If ego or camera died/respawned (e.g., Backspace), re-find and re-attach."""
        if self.ego is None or not self.ego.is_alive:
            self.ego = self._find_ego_by_role(self.world, self.role)

        cam = self._cam_ref() if self._cam_ref else None
        if (self.ego is not None) and (cam is None or not cam.is_alive):
            cam = self._spawn_rgb(self.world, self.ego, self.w, self.h, self.sensor_tick)
            if cam:
                self._cam_ref = weakref.ref(cam)


    @staticmethod
    def _dist2(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))


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
        # 1) Keep ego/cam valid
        self._reacquire_ego_and_camera_if_needed()
        cam = self._cam_ref() if self._cam_ref else None
        if self.ego is None or cam is None:
            # Still not ready; skip this tick gracefully
            return

        # 2) Sync to the world tick (non-blocking-ish)
        snapshot = self.world.wait_for_tick(0.5)
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

        # 4) Update masks via frustum
        if any(x is None for x in (self.center_pts, self.divider_pts, self.bound_pts, self.cross_pts, self.idxes)):
            # Should not happen (we parse once in __init__), but guard anyway
            self._prepare_hdmap_points()

        center_mask = np.zeros(len(self.center_pts), dtype=bool)
        divider_mask = np.zeros(len(self.divider_pts), dtype=bool)
        bound_mask = np.zeros(len(self.bound_pts), dtype=bool)
        cross_mask = np.zeros(len(self.cross_pts), dtype=bool)

        center_mask, divider_mask, bound_mask, cross_mask = ga.filter_by_cameras(
            camera=cam,
            ego_vehicle=ego_snap,
            center_pts=self.center_pts,
            divider_pts=self.divider_pts,
            bound_pts=self.bound_pts,
            cross_pts=self.cross_pts,
            masks=(center_mask, divider_mask, bound_mask, cross_mask),
            max_dist=self.max_dist,
            min_dist=0.1,
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


    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def tick(self):
        """Call this every control cycle to update masks and actor states."""
        self._update_hdmap_info()
        self._update_waypoint_cursor()


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


    def is_traffic_light_red(self):
        # Check if the traffic light in front of the ego vehicle is red
        pass





