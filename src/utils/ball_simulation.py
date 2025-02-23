import warnings

import numpy as np


class State:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy


class BallSim:
    def __init__(self, g, k_over_m, dt):
        self._g = g
        self._k_over_m = k_over_m
        self._dt = dt

    def simulate_euler_step(self, s):
        """Simulate a single step using Euler's method with in-place updates."""
        g, k_over_m, dt = self._g, self._k_over_m, self._dt

        # Compute initial accelerations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            v = np.sqrt(s.vx ** 2 + s.vy ** 2)

            # Check if a warning was issued
            if w:
                for warning in w:
                    if issubclass(warning.category, RuntimeWarning):
                        print(f"RuntimeWarning: Failed to calculate velocity for "
                              f"({s.vx}, {s.vy}) with last position ({s.x}, {s.y})")
                        v = 0  # Handle invalid velocity as needed

        ax = -k_over_m * s.vx * v
        ay = -g - k_over_m * s.vy * v

        # Update velocities
        vx_new = s.vx + ax * dt
        vy_new = s.vy + ay * dt

        # Update positions
        x_new = s.x + s.vx * dt
        y_new = s.y + s.vy * dt

        # Update the state in place
        s.update(x_new, y_new, vx_new, vy_new)

    def simulate_rk2_step(self, s):
        """Simulate a single step using the second-order Runge-Kutta method with in-place updates."""
        g, k_over_m, dt = self._g, self._k_over_m, self._dt

        # Compute initial accelerations
        v = np.sqrt(s.vx ** 2 + s.vy ** 2)
        ax = -k_over_m * s.vx * v
        ay = -g - k_over_m * s.vy * v

        # Predict midpoint velocities
        vx_mid = s.vx + ax * dt / 2
        vy_mid = s.vy + ay * dt / 2

        # Predict midpoint position
        x_mid = s.x + s.vx * dt / 2
        y_mid = s.y + s.vy * dt / 2

        # Compute accelerations at the midpoint
        v_mid = np.sqrt(vx_mid ** 2 + vy_mid ** 2)
        ax_mid = -k_over_m * vx_mid * v_mid
        ay_mid = -g - k_over_m * vy_mid * v_mid

        # Update velocities using midpoint accelerations
        vx_new = s.vx + ax_mid * dt
        vy_new = s.vy + ay_mid * dt

        # Update positions using midpoint velocities
        x_new = s.x + vx_mid * dt
        y_new = s.y + vy_mid * dt

        # Update the state in place
        s.update(x_new, y_new, vx_new, vy_new)
