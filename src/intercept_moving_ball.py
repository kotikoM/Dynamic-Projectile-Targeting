import cv2

from src.utils.ball_simulation import BallSim, State
from src.utils.metrics_estimation import estimate_g_and_k_over_m_with_shooting_method, \
    estimate_velocity_with_shooting_method
from src.utils.visualization import plot_trajectories_for_debugging, absolute_cinema, plot, \
    plot_predicted_centers_against_data, plot_all


def extract_centers(cap, min_radius=3):
    # Initialize background subtractor and video capture
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    centers = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break

        cv2.imshow("Original Frame", frame)

        # Convert to grayscale and apply background subtraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = back_sub.apply(gray)

        # Thresholding and noise removal
        _, thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or frame_count < 2:
            continue

        # Get the largest contour and calculate the enclosing circle
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        # Invert y axis
        if radius > min_radius:
            center = (int(x), frame.shape[1] - int(y))  # Invert y-axis
            centers.append(center)

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red dot

        # Display the processed frame
        cv2.imshow("Frame with Overlay", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return centers


def calculate_velocities(centers, dt):
    velocities = []
    for i in range(1, len(centers)):
        # v = (curr_center - prev_center) / dt
        cx, cy = centers[i]
        px, py = centers[i - 1]

        vx = (cx - px) / dt
        vy = (cy - py) / dt
        velocities.append((vx, vy))
    return velocities


def get_predicted_centers(pos, v, simulate, steps):
    predicted_centers = []
    velocities = []
    x0, y0 = pos
    vx0, vy0 = v
    s = State(x0, y0, vx0, vy0)
    for _ in range(steps):
        predicted_centers.append((s.x, s.y))
        velocities.append((s.vx, s.vy))
        simulate(s)

    return predicted_centers, velocities


if __name__ == '__main__':
    path = "../videos/ball_sim_generated.mp4"

    # Read first how many frames?
    frame_limit = 100

    # how many extra steps to predict
    extra_steps = 100

    # Numerical method
    method = 'euler'

    # Initialize shooter and interception step
    shooter = (100, 300)
    intercept_after_stop = 50

    assert intercept_after_stop < extra_steps, "Intercept step should be less than predicted extra steps"
    # Retrieve video properties
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Properties: FPS={fps}, Total Frames={total_frames}, Resolution={width}x{height}")

    all_centers = extract_centers(cap)
    plot(all_centers)
    centers = all_centers[:frame_limit]
    print(f'Extracted  {len(centers)} centers')

    dt = 1 / fps
    velocities = calculate_velocities(centers, dt)

    initial_steps = len(centers)
    g, k_over_m = estimate_g_and_k_over_m_with_shooting_method(velocities, centers, dt, initial_steps)
    print(f"Estimated g: {g}, k/m: {k_over_m}, with dt: {dt}")

    # Predict path
    if method == 'euler':
        simulate = BallSim(g, k_over_m, dt).simulate_euler_step
    elif method == 'rk2':
        simulate = BallSim(g, k_over_m, dt).simulate_rk2_step
    predicted_centers, predicted_velocities = get_predicted_centers(centers[0], velocities[0], simulate,
                                                                    steps=initial_steps + extra_steps)
    plot_predicted_centers_against_data(all_centers, predicted_centers, frame_limit)

    # Set up shooter
    intercept_step = initial_steps + intercept_after_stop
    target = predicted_centers[intercept_step]
    velocities, max_steps = estimate_velocity_with_shooting_method(shooter, target, simulate, 0.1, intercept_after_stop)
    plot_trajectories_for_debugging(shooter, target, velocities, max_steps, simulate)

    shooter_v = velocities[-1]
    second_ball_centers, second_ball_velocities = get_predicted_centers(shooter, shooter_v, simulate, max_steps + 1)
    plot_all(centers, predicted_centers, second_ball_centers)

    absolute_cinema(width, height, predicted_centers, predicted_velocities, second_ball_centers, second_ball_velocities,
                    intercept_step, g, k_over_m)
