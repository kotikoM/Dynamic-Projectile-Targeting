from src.utils.ball_simulation import BallSim
from src.utils.object_detection import TargetExtractor
from src.utils.metrics_estimation import estimate_velocity_with_shooting_method_dynamically
from src.utils.visualization import plot_trajectories_for_debugging, plot_all_target_trajectories, animate

import time as t

if __name__ == '__main__':
    start = t.time()
    # Image path
    big_batch = "../images/colorful_balls_large_batch.jpg"
    small_batch = "../images/colorful_balls_small_batch.jpg"
    ball_15 = "../images/colorful_balls.jpg"
    path = ball_15

    # Image dimensions
    width = 400
    height = 400

    # Threshold for pixel intensity
    threshold = 200

    # Simulation parameters
    g = 9.8
    k_over_m = 0.02
    dt = 0.01

    if 0 > dt > 2 * (1 / k_over_m):
        raise ValueError("Time step too large for simulation")

    # Tolerance for velocity estimation
    tolerance = 1

    # Shooter Position
    shooter = (200, 200)

    # Numerical method ("euler" or "rk2")
    method = "rk2"

    # Mode ("visualize" or "none")
    mode = "visualize"

    # Initialize simulator
    sim = BallSim(g, k_over_m, dt)

    # Choose simulation method
    if method == "euler":
        simulate = sim.simulate_euler_step
    elif method == "rk2":
        simulate = sim.simulate_rk2_step
    else:
        raise ValueError("Invalid simulation method. Choose 'euler' or 'rk2'.")

    # targets = [(center_x, center_y, radius), ...]
    targets = TargetExtractor(path, width, height, mode).extract(threshold)
    print(f'Found targets in {t.time() - start} seconds')
    print(f"Extracted centers and radii for {len(targets)} balls.")

    # Sort for visually appealing ordering
    targets = sorted(targets, key=lambda t: ((t[0] - shooter[0]) ** 2 + (t[1] - shooter[1]) ** 2) ** 0.5)

    # Estimate velocities
    optimal_velocities_and_time = []
    for tx, ty, r in targets:

        t_velocities, max_time = estimate_velocity_with_shooting_method_dynamically(shooter, (tx, ty), simulate,
                                                                                    tolerance)

        if mode == 'visualize':
            plot_trajectories_for_debugging(shooter, (tx, ty), t_velocities, max_time, simulate)

        optimal_velocities_and_time.append((t_velocities[-1], max_time))

    plot_all_target_trajectories(shooter, targets, optimal_velocities_and_time, simulate)
    print(f'Successfully found targets and their velocities in {(t.time() - start):.2f} seconds.')
    animate(width, height, shooter, targets, optimal_velocities_and_time, simulate)
