"""
Traffic Light & Stop-Sign Detection Module using YOLO.

- detect_traffic_lights(): low-level YOLO boxes with class, color, and confidence.
- get_high_level_signals(): fuses YOLO + LiDAR into 4 signals:

    tl_red: True if a red light affecting our lane is present
    stop_sign_present: True if a stop sign affecting our lane is present
    tl_distance: distance [m] to the controlling red light (if any)
    stop_sign_distance: distance [m] to the controlling stop sign (if any)

Higher-level filtering (confidence, approximate lane relevance via image center)
is done here, so downstream code only sees already-filtered signals.
"""


import cv2
import numpy as np
from ultralytics import YOLO
import math

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


class TrafficLightDetector:
    
    def __init__(self,config ,model_path='./models/yolov8n.pt', use_cuda=True):
        
        self.use_yolo = False
        self.yolo_model = None
        self.config = config

        self.traffic_light_classes = {
            0: 'Red',
            1: 'Yellow', 
            2: 'Green',
            3: 'Sign'
        }
        
        try:
            self.yolo_model = YOLO(model_path)
            device = 'cuda' if use_cuda and self._is_cuda_available() else 'cpu'
            self.yolo_model.to(device)
            self.use_yolo = True
            model_type = "Custom Traffic Light State"
            print(f"[YOLO] Traffic light detection enabled with {model_type} model: {model_path} on {device}")
            print(f"[YOLO] Using custom classes: {self.traffic_light_classes}")
        except Exception as e:
            print(f"[YOLO] Failed to initialize: {e}")
            self.use_yolo = False
    
    def _is_cuda_available(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect_traffic_lights(self, rgb_image):
        if not self.use_yolo or self.yolo_model is None:
            return []
        
        try:
            h_img, w_img = rgb_image.shape[:2]
                        
            results = self.yolo_model(
                rgb_image, 
                verbose=False, 
                imgsz=1280,           # Higher resolution for distant objects
                conf=0.25,            # Higher confidence to reduce false positives
                iou=0.4,              # Higher IoU threshold
                max_det=100            # Allow more detections for custom model
            )
            
            detections = []
            for r in results:
                boxes = r.boxes  # Detections object
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if this is a valid traffic light detection
                    is_valid = False
                    color = 'unknown'
                    
                    if cls in self.traffic_light_classes:
                        is_valid = True
                        color = self.traffic_light_classes[cls].lower()
                    
                    if not is_valid:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Use coordinates directly (no ROI conversion needed)
                    x1_full = int(x1)
                    y1_full = int(y1)
                    x2_full = int(x2)
                    y2_full = int(y2)
                    
                    bbox = [x1_full, y1_full, x2_full, y2_full]
                    bbox_width = x2_full - x1_full
                    bbox_height = y2_full - y1_full
                    
                    # Filter out unrealistic aspect ratios
                    aspect_ratio = bbox_height / max(bbox_width, 1)

                    # Filter out very large bounding boxes (likely not traffic lights)
                    bbox_area = bbox_width * bbox_height

                    # Calculate center position in original image
                    center_x = (x1_full + x2_full) / 2
                    center_y = (y1_full + y2_full) / 2
                    
                    # Prioritize detections closer to image center (horizontally)
                    horizontal_center_distance = abs(center_x - w_img / 2) / (w_img / 2)
                    
                    # Boost confidence for center detections, reduce for edge detections
                    position_weight = 1.0 - (horizontal_center_distance * 0.3)
                    adjusted_conf = conf * position_weight
                    
                    detections.append({
                        'bbox': bbox,
                        'conf': float(adjusted_conf),  # Use position-adjusted confidence
                        'original_conf': float(conf),   # Keep original for debugging
                        'class': cls,
                        'class_name': self.traffic_light_classes[cls],
                        'color': color,
                        'center_distance': float(horizontal_center_distance),
                        'aspect_ratio': float(aspect_ratio)
                    })
            
            # Sort by adjusted confidence to prioritize center and most confident detections
            detections.sort(key=lambda x: x['conf'], reverse=True)
            
            # Filter: keep only top 100 most confident detections to reduce false positives
            detections = detections[:100]
            
            return detections
        
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return []
    

    
    def get_traffic_light_distance(self, bbox, lidar_points, rgb_image, cam_pos, cam_rot, lidar_pos, lidar_rot):

        h, w, _ = rgb_image.shape
        hfov = float(self.config.camera_fov)  # deg
        
        # LiDAR points (N, 3) in LiDAR frame
        pts_lidar = lidar_points[:,:3].astype(np.float32)
        
        # LiDAR pose in ego frame
        lidar_pos = np.array(lidar_pos, dtype=np.float32)       # [x,y,z] in ego
        lidar_roll, lidar_pitch, lidar_yaw = lidar_rot  # deg
        
        Rx_l, Ry_l, Rz_l = _rpy_deg_to_RxRyRz(lidar_roll, lidar_pitch, lidar_yaw)
        R_lidar_to_ego = (Rz_l @ Ry_l @ Rx_l)  # lidar orientation relative to ego
        
        # pts_ego = (pts_lidar @ R_lidar_to_ego.T) + lidar_pos  # (N,3) in ego coords
        pts_ego = R_lidar_to_ego @ (pts_lidar + lidar_pos)  # (N,3) in ego coords

        cam_pos = np.array(cam_pos, dtype=np.float32)       # [x,y,z] in ego
        cam_roll, cam_pitch, cam_yaw = cam_rot                    # deg

        fx, fy, cx, cy = _cam_intrinsics(w, h, hfov)
        R_inv, t = _ego_to_cam_matrix(cam_pos, (cam_roll, cam_pitch, cam_yaw))
        
        distances = []
        
        x1, y1, x2, y2 = bbox

        for p_ego in pts_ego:
            # ego -> camera
            p_rel = p_ego - t
            p_cam = R_inv @ p_rel
            X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

            if X <= 0.05:
                continue

            u = int(cx + fx * (Y / X))
            v = int(cy - fy * (Z / X))

            # 이미지 안 & bbox 안에 있는 포인트만 선택
            if not (0 <= u < w and 0 <= v < h):
                continue
            
            if not (x1 <= u <= x2  and y1  <= v <= y2 ):
                continue

            # 여기서는 "차량 중심 기준 거리"를 사용 (원하면 카메라 기준으로 바꿔도 됨)
            x,y,z = p_cam
            d = float(np.linalg.norm([x,y]))  # 3D distance from ego origin
            distances.append(d)

        if not distances:
            return None

        # 최소값사용
        return float(np.min(distances))



    def get_high_level_signals(
        self,
        detections,
        lidar_points,
        rgb_image,
        cam_pos,
        cam_rot_rpy_deg,
        lidar_pos,
        lidar_rot_rpy_deg,
        min_conf: float = 0.35,
        max_center_distance: float = 0.5,
    ):
        """
        Collapse raw YOLO detections + LiDAR into 4 high-level signals:

            - tl_red: True if a red light affecting our lane is present
            - stop_sign_present: True if a stop sign affecting our lane is present
            - tl_distance: distance [m] to the controlling red light (if any)
            - stop_sign_distance: distance [m] to the controlling stop sign (if any)

        "Relevant" objects are:
            * detection class in {red light, sign}
            * confidence >= min_conf (after the position-based adjustment)
            * horizontally close to image center (center_distance <= max_center_distance)
        """
        tl_red = False
        stop_sign_present = False
        tl_distance = None
        stop_sign_distance = None

        if detections is None or len(detections) == 0:
            return tl_red, stop_sign_present, tl_distance, stop_sign_distance

        if lidar_points is None or len(lidar_points) == 0 or rgb_image is None:
            # No distance possible, but we can still return booleans based on classes
            for det in detections:
                color = str(det.get("color", "")).lower()
                conf = float(det.get("conf", 0.0))
                center_dist = float(det.get("center_distance", 1.0))
                if conf < min_conf or center_dist > max_center_distance:
                    continue
                if color == "red":
                    tl_red = True
                elif color == "sign":
                    stop_sign_present = True
            return tl_red, stop_sign_present, tl_distance, stop_sign_distance

        for det in detections:
            color = str(det.get("color", "")).lower()
            conf = float(det.get("conf", 0.0))
            center_dist = float(det.get("center_distance", 1.0))

            # Only keep high-confidence, roughly-in-lane detections
            if conf < min_conf or center_dist > max_center_distance:
                continue

            bbox = det["bbox"]

            dist = self.get_traffic_light_distance(
                bbox=bbox,
                lidar_points=lidar_points,
                rgb_image=rgb_image,
                cam_pos=cam_pos,
                cam_rot=cam_rot_rpy_deg,
                lidar_pos=lidar_pos,
                lidar_rot=lidar_rot_rpy_deg,
            )
            if dist is None:
                continue

            if color == "red":
                # Keep the closest red TL
                if tl_distance is None or dist < tl_distance:
                    tl_distance = dist
                    tl_red = True
            elif color == "sign":
                # Treat "Sign" class as stop sign; keep the closest one
                if stop_sign_distance is None or dist < stop_sign_distance:
                    stop_sign_distance = dist
                    stop_sign_present = True

        return tl_red, stop_sign_present, tl_distance, stop_sign_distance
