# offline_hdmap_visualize_headless.py
# Usage:
#   python offline_hdmap_visualize_headless.py --xodr my_map.xodr --out map_hd.png \
#       --wp_step 1.0 --img_w 3000 --img_h 3000 --pt_step 0.5

import os, sys, math, argparse
from collections import defaultdict
import numpy as np

# --- Headless (no GUI) ---
import matplotlib
matplotlib.use("Agg")  # << 중요: GUI 없이 파일 저장만
import matplotlib.pyplot as plt

try:
    import carla
except ImportError:
    print("ERROR: carla module not found. Add CARLA egg to PYTHONPATH.")
    raise

# ----------------------
# 유틸 함수들
# ----------------------
def forward_left_unit(fwd_vec):
    n = math.hypot(fwd_vec.x, fwd_vec.y) or 1.0
    return (-fwd_vec.y / n, fwd_vec.x / n)

def offset_point(loc, fwd_vec, offset):
    lx, ly = forward_left_unit(fwd_vec)
    return (loc.x + lx * offset, loc.y + ly * offset, loc.z)

def sample_linestring(points_xyz, step=0.5):
    if len(points_xyz) < 2:
        return np.asarray(points_xyz, dtype=float)[:, :2] if len(points_xyz) else np.empty((0,2))
    pts = np.asarray(points_xyz, dtype=float)[:, :2]
    segs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(segs, axis=1)
    out = [pts[0]]
    for i, L in enumerate(seg_len):
        if L == 0:
            continue
        n = max(1, int(np.floor(L / step)))
        for k in range(1, n + 1):
            t = min(1.0, k * step / L)
            out.append(pts[i] * (1 - t) + pts[i + 1] * t)
    return np.array(out)

def lane_groups(carla_map, distance=1.0, lane_types=None):
    if lane_types is None:
        lane_types = carla.LaneType.Driving
    wps = [w for w in carla_map.generate_waypoints(distance) if (w.lane_type & lane_types) != carla.LaneType.NONE]
    groups = defaultdict(list)
    for w in wps:
        groups[(w.road_id, w.section_id, w.lane_id)].append(w)
    return groups

def waypoint_polyline(group):
    return [(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in group]

def driving_neighbors_flags(w):
    left = w.get_left_lane()
    right = w.get_right_lane()
    has_left = left is not None and (left.lane_type & carla.LaneType.Driving) != carla.LaneType.NONE
    has_right = right is not None and (right.lane_type & carla.LaneType.Driving) != carla.LaneType.NONE
    return has_left, has_right

def boundary_from_group(group, side="left"):
    coords = []
    for w in group:
        loc = w.transform.location
        fwd = w.transform.get_forward_vector()
        width = w.lane_width
        off = (width / 2.0) * (+1 if side == "left" else -1)
        coords.append(offset_point(loc, fwd, off))
    return coords
def get_crosswalk_lines(carla_map):
    """
    교차로 횡단보도 외곽 polyline들을 반환.
    - CARLA 버전에 따라:
      * List[List[carla.Location]] 또는
      * List[carla.Location] (여러 폐곡선이 한 리스트에 이어짐)
    두 형태 모두를 처리한다.
    """
    res = carla_map.get_crosswalks()
    if not res:
        return []

    rings = []

    # 케이스 A: 리스트의 리스트
    if isinstance(res[0], (list, tuple)):
        for ring in res:
            if not ring:
                continue
            coords = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in ring]
            # 닫혀있지 않으면 닫아줌
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            rings.append(coords)
        return rings

    # 케이스 B: 평평한 리스트 (List[Location]) — 반복되는 시작점을 기준으로 폴리곤 분할
    if hasattr(res[0], 'x'):  # carla.Location
        current = []
        start = None  # (x,y,z)
        for p in res:
            xyz = (p.x, p.y, getattr(p, 'z', 0.0))
            if not current:
                current = [xyz]
                start = xyz
                continue
            current.append(xyz)
            # 시작점과 동일해지면 폴리곤 하나가 닫힘
            if xyz == start and len(current) > 2:
                rings.append(current)
                current = []
                start = None
        # 혹시 마지막이 닫히지 않은 경우 보정
        if current:
            if len(current) > 2:
                if current[0] != current[-1]:
                    current.append(current[0])
                rings.append(current)
        return rings

    # 예상치 못한 타입 방어
    return []

def normalize_to_image(xy_pts, bounds, img_w, img_h, keep_aspect=True, margin=20):
    (minx, miny, maxx, maxy) = bounds
    if keep_aspect:
        sx = (img_w - 2 * margin) / (maxx - minx + 1e-6)
        sy = (img_h - 2 * margin) / (maxy - miny + 1e-6)
        s = min(sx, sy)
        offx = (img_w - s * (maxx - minx)) / 2.0
        offy = (img_h - s * (maxy - miny)) / 2.0
        x = offx + (xy_pts[:, 0] - minx) * s
        y = offy + (xy_pts[:, 1] - miny) * s
    else:
        x = margin + (xy_pts[:, 0] - minx) * (img_w - 2 * margin) / (maxx - minx + 1e-6)
        y = margin + (xy_pts[:, 1] - miny) * (img_h - 2 * margin) / (maxy - miny + 1e-6)
    y = img_h - y
    return np.stack([x, y], axis=1)

def compute_global_bounds(poly_sets):
    xs, ys = [], []
    for polys in poly_sets:
        for pts in polys:
            if len(pts) == 0:
                continue
            arr = np.asarray(pts)[:, :2]
            xs.append(arr[:, 0]); ys.append(arr[:, 1])
    if not xs:
        return (0,0,1,1)
    x = np.concatenate(xs); y = np.concatenate(ys)
    return (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

def plot_points_sets(out_path, img_w, img_h, sets_with_style):
    plt.figure(figsize=(img_w/100.0, img_h/100.0), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("black")
    for arr, color, size, label in sets_with_style:
        if arr is None or len(arr)==0:
            continue
        plt.scatter(arr[:,0], arr[:,1], s=size, c=color, marker='.', linewidths=0, label=label)
    plt.axis('off')
    if any(lbl for _,_,_,lbl in sets_with_style):
        plt.legend(loc='lower left', fontsize=8)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, facecolor='black')  # 파일로 저장만
    plt.close()

# ----------------------
# 메인
# ----------------------
def get_idx_points(pts):
    idx_points = []
    for idx, pt in enumerate(pts):
        idx_points.extend([idx]*len(pt))
    return np.array(idx_points)
def extract_waypoints(xodr_text, wp_step = 1.0, pt_step=0.5):


    # 오프라인 맵 생성
    cmap = carla.Map("offline_map", xodr_text)

    # centerline
    groups = lane_groups(cmap, distance=wp_step, lane_types=carla.LaneType.Driving)
    centerlines = [ waypoint_polyline(g) for g in groups.values() ]

    # divider (내부 경계)
    dividers = []
    for grp in groups.values():
        w = grp[0]
        has_left, has_right = driving_neighbors_flags(w)
        if has_left:
            dividers.append(boundary_from_group(grp, side="left"))
        if has_right:
            dividers.append(boundary_from_group(grp, side="right"))

    # boundary (바깥 경계)
    boundaries = []
    for grp in groups.values():
        w = grp[0]
        has_left, has_right = driving_neighbors_flags(w)
        if not has_left:
            boundaries.append(boundary_from_group(grp, side="left"))
        if not has_right:
            boundaries.append(boundary_from_group(grp, side="right"))

    # crossing
    crossings = get_crosswalk_lines(cmap)

    # 리샘플 (점 구름)
    def resample_all(lines):
        out=[]
        for poly in lines:
            arr = sample_linestring(poly, step=pt_step)
            if len(arr)>0:
                out.append(arr)
        return out

    center_pts = resample_all(centerlines)
    divider_pts = resample_all(dividers)
    bound_pts   = resample_all(boundaries)
    cross_pts   = resample_all(crossings)
    idxes = [get_idx_points(center_pts), get_idx_points(divider_pts), get_idx_points(bound_pts), get_idx_points(cross_pts)]
    center_pts = np.concatenate(center_pts)
    divider_pts = np.concatenate(divider_pts)
    bound_pts = np.concatenate(bound_pts)
    cross_pts = np.concatenate(cross_pts)
    
    center_pts = np.concatenate([center_pts, np.zeros_like(center_pts)[:,0:1]], axis= 1)
    divider_pts = np.concatenate([divider_pts, np.zeros_like(divider_pts)[:,0:1]], axis= 1)
    bound_pts = np.concatenate([bound_pts, np.zeros_like(bound_pts)[:,0:1]], axis= 1)
    cross_pts = np.concatenate([cross_pts, np.zeros_like(cross_pts)[:,0:1]], axis= 1)
    
    return center_pts, divider_pts, bound_pts, cross_pts, idxes
    #-----------------------by youngho---------------#
    import pdb; pdb.set_trace()


    #------------------------------------------------#
    # 이미지 좌표 변환
    bounds = compute_global_bounds([center_pts, divider_pts, bound_pts, cross_pts])

    def to_img_points(list_of_arrays):
        out=[]
        for arr in list_of_arrays:
            xy = normalize_to_image(arr, bounds, args.img_w, args.img_h, keep_aspect=True)
            out.append(xy)
        return out

    center_img = to_img_points(center_pts)
    divider_img = to_img_points(divider_pts)
    bound_img   = to_img_points(bound_pts)
    cross_img   = to_img_points(cross_pts)

    def stack_or_none(list_xy):
        if not list_xy:
            return None
        return np.vstack(list_xy) if len(list_xy)>1 else list_xy[0]

    center_all = stack_or_none(center_img)
    divider_all = stack_or_none(divider_img)
    bound_all   = stack_or_none(bound_img)
    cross_all   = stack_or_none(cross_img)

    # 색상: center(청록), divider(흰), boundary(노랑), crossing(마젠타)
    sets = [
        (center_all,  "#00FFFF", 30, "centerline"),
        (divider_all, "#FFFFFF", 30, "divider"),
        (bound_all,   "#FFD700", 30, "boundary"),
        (cross_all,   "#FF00FF", 30, "crossing"),
    ]
    plot_points_sets(args.out, args.img_w, args.img_h, sets)
    print(f"[OK] saved: {args.out}")
