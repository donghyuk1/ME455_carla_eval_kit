class MyAutoConfig:
    # Camera (front only)
    camera_pos   = [1.3, 0.0, 2.3]
    camera_width = 960
    camera_height = 480
    camera_fov   = 120
    camera_rot_0 = [0.0, 0.0, 0.0]

    # LiDAR
    lidar_pos = [1.3, 0.0, 2.5]
    lidar_rot = [0.0, 0.0, 0.0]

    # Simulator timing
    carla_frame_rate = 1.0 / 20.0
    carla_fps = 20

    # Route planner
    route_planner_min_distance = 4
    route_planner_max_distance = 20.0

    # Control
    target_speed = 12.0  
    steer_gain   = 0.5
    steer_damping = 0.9

    # PID (longitudinal)
    pid_Kp = 0.6
    pid_Ki = 0.0
    pid_Kd = 0.005
    pid_u_min = 0.0                  # lower saturation bound (minimum throttle output)
    pid_u_max = 0.7                  # upper saturation bound (maximum throttle output, prevents over-acceleration)

    # Safety box
    safety_x_min = 1.0               # start of forward safety zone (meters ahead of ego)
    safety_x_max = 10.0               # end of forward safety zone → stop if obstacle within this distance
    safety_y_abs = 1.2               # half-width of safety zone (meters to left/right from centerline)
    safety_z_min = -1.5              # lower vertical bound (filters ground points below car)
    safety_z_max = 1.0               # upper vertical bound (ignores high clutter like trees, signs)

    # Visualization
    show_window = True

    # --- Ego vehicle size --- 
    ego_extent_x = 2.4508416652679443  # half-length of ego vehicle (used for bounding box)
    ego_extent_y = 1.0641621351242065  # half-width of ego vehicle
    ego_extent_z = 0.7553732395172119  # half-height of ego vehicle
