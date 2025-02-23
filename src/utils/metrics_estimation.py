import numpy as np

from src.utils.ball_simulation import State, BallSim


def distance(x, y, tx, ty):
    return np.sqrt((tx - x) ** 2 + (ty - y) ** 2)


def estimate_velocity_with_shooting_method_dynamically(shooter, target, simulate, tolerance, step_factor=5, max_iter=10, h=0.1):
    x, y = shooter
    xt, ty = target
    max_time = int(distance(x, y, xt, ty) * step_factor)
    return estimate_velocity_with_shooting_method(shooter, target, simulate, tolerance, max_time, max_iter, h)


def estimate_velocity_with_shooting_method(shooter, target, simulate, tolerance, max_time, max_iter=10, h=0.1):
    x, y = shooter
    xt, ty = target
    vx, vy = 0, 0
    print(f"Estimating optimal velocity for ({xt:.1f}, {ty:.1f}) from ({x:.1f}, {y:.1f})...")

    velocities = []
    for i in range(max_iter):
        s = State(x, y, vx, vy)
        s1 = State(x, y, vx + h, vy)
        s2 = State(x, y, vx, vy + h)

        # Simulate motion for max_time steps
        for _ in range(max_time):
            simulate(s)
            simulate(s1)
            simulate(s2)

        err = distance(s.x, s.y, xt, ty)
        if err <= tolerance:
            print(f'Found optimal velocity for ({xt:.1f}, {ty:.1f}), with error of {err:.1f}.')
            print(f'Starting velocity ({vx:.1f}, {vy:.1f}), ending velocity ({s.vx:.1f}, {s.vy:.1f}).')
            break

        # Jacobain matrix
        j11, j12 = (s1.x - s.x) / h, (s2.x - s.x) / h
        j21, j22 = (s1.y - s.y) / h, (s2.y - s.y) / h
        e1, e2 = xt - s.x, ty - s.y
        det = j11 * j22 - j12 * j21

        dvx = (j22 * e1 - j12 * e2) / det
        dvy = (-j21 * e1 + j11 * e2) / det

        vx += dvx
        vy += dvy
        velocities.append((vx, vy))

    print("-" * 50)
    return velocities, max_time


def estimate_g_and_k_over_m_with_shooting_method(velocities, centers, dt, max_steps, max_iter=500, tol=1):
    # Unpack
    vx, vy = velocities[0]
    x, y = centers[0]

    def get_errors(g, k_over_m, ):
        sim = BallSim(g, k_over_m, dt).simulate_euler_step
        s = State(x, y, vx, vy)
        tx, ty = centers[-1]
        for i in range(max_steps):
            sim(s)
        return tx - s.x, ty - s.y

    g, k_over_m = 0.0, 0.0
    delta = 1e-5
    for i in range(max_iter):
        ex, ey = get_errors(g, k_over_m)
        if abs(ex) < tol and abs(ey) < tol:
            print(f"Found g: {g}, k/m: {k_over_m}")
            print(f"Converged after {i} iterations.")
            return g, k_over_m

        # Run g with delta
        g_ex, g_ey = get_errors(g + delta, k_over_m)
        g_ex, g_ey = (g_ex - ex) / delta, (g_ey - ey) / delta

        # Run k/m with delta
        km_ex, km_ey = get_errors(g, k_over_m + delta)
        km_ex, km_ey = (km_ex - ex) / delta, (km_ey - ey) / delta

        J = np.array([
            [g_ex, km_ex],
            [g_ey, km_ey]
        ])
        F = np.array([ex, ey])

        d_g, d_km = np.linalg.solve(J, -F)
        g += d_g
        k_over_m += d_km
