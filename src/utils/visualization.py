import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

from src.utils.ball_simulation import State


def plot_dbscan_labels(points, labels, show_axis='on'):
    unique_labels = set(labels)
    plt.figure(figsize=(10, 10))

    # colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colormap = cm.get_cmap("Spectral")  # Choose the Spectral colormap
    colors = [colormap(each) for each in np.linspace(0, 1, len(unique_labels))]

    for label, color in zip(unique_labels, colors):
        if label == -1:  # Plot noise points
            color = 'black'
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label}'

        plt.scatter(points[labels == label][:, 1], points[labels == label][:, 0],
                    s=1, color=color, label=label_name)

    plt.axis(show_axis)
    plt.gca().invert_yaxis()
    plt.title(f'DBSCAN Clustering - {len(np.unique(labels)) - 1} clusters found')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def plot_targets(targets):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Center')
    blue_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Object')

    for center_x, center_y, radius in targets:
        circle = plt.Circle((center_x, center_y), radius, color='blue', ec="black", lw=1.5, alpha=0.5)
        ax.add_patch(circle)

        ax.plot(center_x, center_y, marker='o', color='red', markersize=5)

    ax.legend(handles=[red_dot, blue_circle])

    # Set the plot limits and remove axes
    # ax.axis('off')  # Turn off the axes for a gridless look

    plt.title('Target Centers and Radii')
    plt.show()


def plot_target_trajectory(shooter, target, velocity, width, height, simulate):
    xs, ys = shooter
    xt, yt = target
    vx, vy = velocity

    s = State(xs, ys, vx, vy)
    trajectory_x, trajectory_y = [], []

    while True:
        simulate(s)
        x, y = s.x, s.y

        trajectory_x.append(x)
        trajectory_y.append(y)

        if x < 0 or x > width or y < 0:
            break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Ball Trajectory")

    ax.plot(xs, ys, 'go', label="Shooter", markersize=10)  # Shooter (green circle)
    ax.plot(xt, yt, 'ro', label="Target", markersize=10)  # Target (red circle)

    ax.plot(trajectory_x, trajectory_y, 'b-', label="Trajectory")  # Ball trajectory (blue line)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_trajectories_for_debugging(shooter, target, velocities, steps, simulate):
    xs, ys = shooter
    xt, yt = target

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title(f"Ball Trajectory")
    ax.plot(xs, ys, 'go', label="Shooter", markersize=10)  # Shooter (green circle)
    ax.plot(xt, yt, 'ro', label=f"Target ({xt:.1f}, {yt:.1f})", markersize=10)  # Target (red circle)

    colormap = cm.get_cmap("rainbow", len(velocities))

    for i, (vx, vy) in enumerate(velocities):
        trajectory_x, trajectory_y = [], []
        s = State(xs, ys, vx, vy)
        for _ in range(steps):
            simulate(s)
            x, y = s.x, s.y
            trajectory_x.append(x)
            trajectory_y.append(y)

        # Assign a unique color and label to each trajectory
        ax.plot(
            trajectory_x,
            trajectory_y,
            color=colormap(i),
            label=f"Path {i + 1} v({vx:.2f}, {vy:.2f})"
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_all_target_trajectories(shooter, targets, velocities, simulate):
    xs, ys = shooter

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title(f"Ball Trajectories")
    ax.plot(xs, ys, 'go', label="Shooter", markersize=10)
    colormap = cm.get_cmap("rainbow", len(velocities))

    for i in range(len(targets)):
        tx, ty, r = targets[i]
        (vx, vy), steps = velocities[i]

        trajectory_x, trajectory_y = [], []
        s = State(xs, ys, vx, vy)
        for _ in range(steps):
            simulate(s)
            trajectory_x.append(s.x)
            trajectory_y.append(s.y)

        print(f"For target {i + 1}, trajectory length: {len(trajectory_x)}")

        ax.plot(tx, ty, 'o', color=colormap(i), label="Target", markersize=10)
        # ax.text(tx, ty - r - 2, f"Target {i + 1}", ha='center', va='top', fontsize=8, color=colormap(i))
        ax.plot(trajectory_x, trajectory_y, color=colormap(i), label=f"Target {i + 1} ({round(tx)}, {round(ty)})")

    # ax.legend()
    # ax.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_all(centers, predicted_centers, second_ball_centers):
    x_coords, y_coords = zip(*centers)
    x_pred, y_pred = zip(*predicted_centers)
    x_second, y_second = zip(*second_ball_centers)

    plt.figure(figsize=(8, 6))
    plt.plot(x_pred, y_pred, 'o-', label="Prediction", color='red', alpha=0.5, markersize=3)
    plt.plot(x_coords, y_coords, 'o-', label="Data", color='blue', alpha=0.5, markersize=3)
    plt.plot(x_second, y_second, 'o-', label="Second Ball", color='green')
    plt.title("Object Center Trajectory")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend()
    plt.grid()
    plt.show()


def plot(centers):
    x_coords, y_coords = zip(*centers)

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, 'o-', label="Object Center", color='blue')
    plt.title("Object Center Trajectory")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_predicted_centers_against_data(all_centers, predicted_centers, frame_limit):
    x_pred, y_pred = zip(*predicted_centers)
    x_coords, y_coords = zip(*all_centers)
    x_highlight, y_highlight = zip(*all_centers[:frame_limit])

    plt.figure(figsize=(8, 6))

    plt.plot(x_pred, y_pred, 'o-', label="Prediction", color='red', alpha=0.5, markersize=3)
    plt.plot(x_coords, y_coords, 'o-', label="Data", color='blue', alpha=0.5, markersize=3)
    plt.plot(x_highlight, y_highlight, 'x', label="Highlighted (First Frame Limit)", color='green', markersize=6)
    plt.title("Object Center Trajectory")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend()
    plt.grid()
    plt.show()


def animate(width, height, shooter, targets, velocities, simulate, scale=2, output_file="simulation.mp4", fps=60):
    shooter_bgr = (41, 4, 217)

    target_bgr = (174, 153, 141)
    hit_target_bgr = (66, 45, 43)

    projectile_bgr = (244, 242, 327)
    trajectory_bgr = (60, 35, 239)

    text_bgr = (244, 242, 327)

    # Apply scaling to dimensions
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (scaled_width, scaled_height))

    canvas = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)

    # Scale shooter position
    xs, ys = shooter
    xs, ys = int(xs * scale), int(ys * scale)
    ys_screen = scaled_height - ys  # Flip vertical axis for visualization
    cv2.circle(canvas, (xs, ys_screen), 10, shooter_bgr, -1)

    # Draw targets
    for tx, ty, r in targets:
        tx, ty, r = int(tx * scale), int(ty * scale), int(r * scale)
        ty_screen = scaled_height - ty
        cv2.circle(canvas, (tx, ty_screen), r, target_bgr, -1)

    # Draw trajectories
    for i in range(len(targets)):
        tx, ty, r = targets[i]
        tx, ty, r = int(tx * scale), int(ty * scale), int(r * scale)
        ty_screen = scaled_height - ty
        (vx, vy), steps = velocities[i]

        trajectory_x, trajectory_y = [], []
        velocities_x, velocities_y = [], []
        s = State(xs / scale, ys / scale, vx, vy)  # Scale back to original coordinates

        # Calculate trajectory
        for _ in range(steps):
            simulate(s)
            trajectory_x.append(s.x * scale)
            velocities_x.append(s.vx)
            trajectory_y.append(s.y * scale)
            velocities_y.append(s.vx)

        skip = 3
        for step in range(0, steps, skip):
            frame = canvas.copy()

            # Draw the entire trajectory
            for px, py in zip(trajectory_x[:step + 1], trajectory_y[:step + 1]):
                py_screen = scaled_height - py
                cv2.circle(frame, (int(px), int(py_screen)), 2, trajectory_bgr, -1)

            # Save previous frame
            canvas = frame.copy()

            # Draw the current position of the projectile
            py_screen = scaled_height - trajectory_y[step]
            cv2.circle(frame, (int(trajectory_x[step]), int(py_screen)), 5, projectile_bgr, -1)

            # Display velocity information
            vx, vy = velocities_x[step], velocities_y[step]
            text_vx = f"vx: {vx:.2f}"
            text_vy = f"vy: {vy:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text_vx, (int(trajectory_x[step]), int(py_screen) + 10), font, 0.5, text_bgr, 1,
                        cv2.LINE_AA)
            cv2.putText(frame, text_vy, (int(trajectory_x[step]), int(py_screen) + 30), font, 0.5, text_bgr, 1,
                        cv2.LINE_AA)

            video_writer.write(frame)

            cv2.imshow("Simulation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                video_writer.release()
                return

        # Turn hit target other color
        cv2.circle(canvas, (tx, ty_screen), r, hit_target_bgr, -1)
        print(f"Target {i + 1} reached at ({tx // scale}, {ty // scale})")

    video_writer.release()
    cv2.destroyAllWindows()


def absolute_cinema(width, height, ball1_path, ball1_velocities, ball2_path, ball2_velocities, intercept_step, g, k_over_m,
                    output_file="simulation.mp4", fps=20):
    # Initialize video writer
    n_height = np.max([y for x, y in ball1_path])
    height = max(height, int(n_height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_bgr = (255, 255, 255)  # White color
    thickness = 1

    # Adjust the paths to match in length
    ball1_path = ball1_path[:intercept_step + 1]
    ball1_velocities = ball1_velocities[:intercept_step + 1]

    padding = [ball2_path[0]] * (len(ball1_path) - len(ball2_path))
    ball2_path = padding + ball2_path

    v_padding = [(0, 0)] * (len(ball1_velocities) - len(ball2_velocities))
    ball2_velocities = v_padding + ball2_velocities
    assert len(ball1_path) == len(ball2_path)

    ball1_radius = 10
    ball2_radius = 10

    # Define color
    ball1_color_bgr = (41, 4, 217)
    path1_color_bgr = (135, 135, 135)
    ball2_color_bgr = (29, 89, 20)
    path2_color_bgr = (56, 170, 153)

    for i in range(1, len(ball1_path)):
        frame = canvas.copy()
        b1x, b1y = ball1_path[i]
        b2x, b2y = ball2_path[i]
        b1_vx, b1_vy = ball1_velocities[i]
        b2_vx, b2_vy = ball2_velocities[i]

        b1x, b2x = int(b1x), int(b2x)
        b1y, b2y = int(height - b1y), int(height - b2y)

        prev_b1x, prev_b1y = int(ball1_path[i - 1][0]), int(height - ball1_path[i - 1][1])
        prev_b2x, prev_b2y = int(ball2_path[i - 1][0]), int(height - ball2_path[i - 1][1])

        # Draw ball1 trajectories
        cv2.line(frame, (prev_b1x, prev_b1y), (b1x, b1y), path1_color_bgr, 2)
        cv2.line(frame, (prev_b2x, prev_b2y), (b2x, b2y), path2_color_bgr, 2)

        canvas = frame.copy()

        # Draw positions
        cv2.circle(frame, (b1x, b1y), ball1_radius, ball1_color_bgr, -1)
        cv2.circle(frame, (b2x, b2y), ball2_radius, ball2_color_bgr, -1)

        # Put velocity text
        b1_vx_text , b1_vy_text = f"vx: {b1_vx:.2f}", f"vy: {b1_vy:.2f}"
        cv2.putText(frame, b1_vx_text, (b1x, b1y - 10), font, 0.5, color_bgr, thickness, cv2.LINE_AA)
        cv2.putText(frame, b1_vy_text, (b1x, b1y - 30), font, 0.5, color_bgr, thickness,cv2.LINE_AA)

        b2_vx_text, b2_vy_text = f"vx: {b2_vx:.2f}", f"vy: {b2_vy:.2f}"
        cv2.putText(frame, b2_vx_text, (b2x, b2y - 10), font, 0.5, color_bgr, thickness, cv2.LINE_AA)
        cv2.putText(frame, b2_vy_text, (b2x, b2y - 30), font, 0.5, color_bgr, thickness, cv2.LINE_AA)

        # Overlay 'g' and 'k/m' as separate text lines
        cv2.putText(canvas, f"g : {g:.2f}", (10, height // 3), font, 0.5, color_bgr, thickness, cv2.LINE_AA)  # 'g' at the top
        cv2.putText(canvas, f"k/m : {k_over_m:.3f}", (10, height // 3 + 20), font, 0.5, color_bgr, thickness, cv2.LINE_AA)  # 'k/m' below 'g'

        video_writer.write(frame)

        cv2.imshow("Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video_writer.release()
            return

    video_writer.release()
    cv2.destroyAllWindows()
