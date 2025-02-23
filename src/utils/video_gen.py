import cv2
import numpy as np

from src.intercept_moving_ball import plot
from src.utils.ball_simulation import State, BallSim

# Simulation state
x, y = 0, 200
vx, vy = 200, 20
s = State(x, y, vx, vy)

# Simulation parameters
g = 20
k_over_m = 0.01
fps = 30
dt = 1 / fps

simulate = BallSim(g, k_over_m, dt).simulate_rk2_step

# Video settings with increased pixel density
base_width, base_height = 300, 300  # Logical simulation area
scaling_factor = 4  # Scale up the resolution
frame_width = base_width * scaling_factor
frame_height = base_height * scaling_factor

output_path = "ball_sim_generated_down.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# White canvas with higher resolution
canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

# Path storage
path = []
for _ in range(200):
    path.append((s.x, s.y))
    simulate(s)
    if s.y < 0:
        break

print(len(path))
plot(path)

# Normalize and scale coordinates for higher resolution
def normalize_coords(x, y):
    ball_x = int(x * scaling_factor)
    ball_y = int(frame_height - y * scaling_factor)  # Flip y-axis
    return ball_x, ball_y

# Draw path onto high-resolution canvas
for x, y in path:
    frame = canvas.copy()

    ball_x, ball_y = normalize_coords(x, y)
    cv2.circle(frame, (ball_x, ball_y), 5 * scaling_factor, (102, 103, 78), -1)

    video_writer.write(frame)

# Release resources
video_writer.release()
print(f"High-resolution video saved to {output_path}")
