import argparse
import os
import time
import weakref

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

    # Attach our own RGB sensor (invisible window; only for frustum filtering)
    cam = _spawn_rgb(world, ego, w, h)
    if cam is None:
        print("[hdmap_vis] Failed to spawn RGB sensor. Retrying with a different pose...")
        cam = _spawn_rgb(world, ego, w, h, loc=(0.5, 0.1, 2.2), rot=(-8.0, 0.0, 0.0))
    if cam is None:
        print("[hdmap_vis] Could not spawn any RGB sensor. Exiting.")
        return

    # We’ll grab snapshots from world; image stream isn’t strictly required for this vis
    cam_ref = weakref.ref(cam)

    print("[hdmap_vis] Running. Press Ctrl+C to quit.")
    frame_idx = 0

    try:
        while True:
            # Keep ego alive (e.g., if player respawned in manual_control)
            if ego is None or ego.is_alive is False:
                # try to re-acquire by role_name
                ego = _find_ego_by_role(world, args.role)
                if ego and (cam_ref() is None or cam_ref().is_alive is False):
                    cam = _spawn_rgb(world, ego, w, h)
                    cam_ref = weakref.ref(cam)

            snapshot = world.wait_for_tick(2.0)
            if snapshot is None:
                continue  # keep trying

            # Get ego + others snapshots
            actors = world.get_actors()
            if ego is None:
                # nothing to do this tick
                continue

            try:
                ego_snap = cva.snap_processing([ego], snapshot)[0]
            except Exception:
                continue

            vehicles = [a for a in actors.filter("vehicle.*") if a.id != ego.id]
            walkers  = list(actors.filter("walker.*"))
            vehicle_snaps = cva.snap_processing(vehicles, snapshot) if vehicles else []
            walker_snaps  = cva.snap_processing(walkers,  snapshot) if walkers  else []

            # Build masks via camera frustum
            center_mask  = np.zeros(len(center_pts),  dtype=bool)
            divider_mask = np.zeros(len(divider_pts), dtype=bool)
            bound_mask   = np.zeros(len(bound_pts),   dtype=bool)
            cross_mask   = np.zeros(len(cross_pts),   dtype=bool)

            camera_actor = cam_ref()
            if camera_actor is None or camera_actor.is_alive is False:
                # Respawn the camera if needed
                cam = _spawn_rgb(world, ego, w, h)
                camera_actor = cam
                cam_ref = weakref.ref(cam)
                if camera_actor is None:
                    continue

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

            # 3D bboxes
            veh_bboxes3d  = ga.get_vehicle_bbox(ego_snap, vehicle_snaps, radius=args.dist)
            walk_bboxes3d = ga.get_vehicle_bbox(ego_snap, walker_snaps,  radius=args.dist)

            # Compose the visualization
            vis = ga.get_gemap_vis(
                center_pts[center_mask],
                divider_pts[divider_mask],
                bound_pts[bound_mask],
                cross_pts[cross_mask],
                (veh_bboxes3d, walk_bboxes3d),
                ego_snap,
                camera_units=[{"name": "RGB0", "sensor": camera_actor, "depth_sensor": None, "dirs": {}}]
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