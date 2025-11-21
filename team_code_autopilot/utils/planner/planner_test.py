# A simple test of the A* planner on a 10x10 grid with some blocked nodes
import matplotlib.pyplot as plt
import numpy as np
from planner import AStarPlanner

# --- Base 10x10 grid of points ---
xs = np.arange(10)
ys = np.arange(10)
grid_x, grid_y = np.meshgrid(xs, ys)
full_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

# ------------------------------------------------------
# Make the environment harder:
#   - Block a vertical "wall" around x = 5
#   - Leave a small gap so that a path still exists
# ------------------------------------------------------
blocked_mask = np.zeros(len(full_points), dtype=bool)

# Helper to get index of a grid coordinate (x, y)
def idx(x, y):
    return y * 10 + x  # because we used meshgrid(xs, ys)

# Block a wall at x=5, for y=1..8 except a gap at y=4
for y_val in range(1, 9):
    if y_val == 4:  # gap
        continue
    blocked_mask[idx(5, y_val)] = True

# You can add a second wall if you want it even harder, e.g.:
# for x_val in range(3, 9):
#     if x_val == 7:  # gap
#         continue
#     blocked_mask[idx(x_val, 6)] = True

# Make sure start and goal are NOT blocked
start_xy = np.array([2.0, 2.0], dtype=np.float32)
goal_xy  = np.array([8.0, 8.0], dtype=np.float32)
blocked_mask[idx(int(start_xy[0]), int(start_xy[1]))] = False
blocked_mask[idx(int(goal_xy[0]),  int(goal_xy[1]))] = False

# Points used by the planner = only non-blocked nodes
free_mask = ~blocked_mask
points = full_points[free_mask]

# --- Create planner with neighbor radius slightly larger than grid spacing ---
planner = AStarPlanner(points=points, neighbor_radius=1.5)

# Plan (note: planner will snap start/goal to the closest free nodes)
path_xy = planner.plan(start_xy, goal_xy)

print("Start:", start_xy)
print("Goal:", goal_xy)
print("Path length:", len(path_xy) if path_xy is not None else None)
print("Path points:\n", path_xy)

# --- Visualization ---
plt.figure()

# Plot all free points
plt.scatter(points[:, 0], points[:, 1], marker='o', label="Free nodes")

# Plot blocked points (obstacles)
blocked_points = full_points[blocked_mask]
if len(blocked_points) > 0:
    plt.scatter(blocked_points[:, 0], blocked_points[:, 1],
                marker='x', label="Blocked nodes")

# Plot the path
if path_xy is not None:
    plt.plot(path_xy[:, 0], path_xy[:, 1], marker='o', linewidth=2, label="A* path")
    plt.text(start_xy[0] + 0.1, start_xy[1] + 0.1, "S")
    plt.text(goal_xy[0] + 0.1,  goal_xy[1]  + 0.1, "G")
else:
    print("No path found! (graph disconnected)")

plt.title("A* on 10x10 Grid with Obstacles (2,2) -> (8,8)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()