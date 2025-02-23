import matplotlib.pyplot as plt
import numpy as np
import random


def generate_smaller_dispersed_circles():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.1, 1.1)  # Extend the range for more dispersion
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axes for a gridless look

    num_balls = 10  # Increase the number of balls
    separation_buffer = 0.02  # Buffer distance for visible separation
    balls = []

    def is_overlapping(x, y, radius, existing_balls):
        for bx, by, br in existing_balls:
            distance = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
            if distance < radius + br + separation_buffer:
                return True
        return False

    for _ in range(num_balls):
        while True:
            x, y = random.uniform(0, 1), random.uniform(0, 1)
            radius = random.uniform(0.02, 0.05)  # Smaller radius for smaller circles
            if not is_overlapping(x, y, radius, balls):
                balls.append((x, y, radius))
                break

    for x, y, radius in balls:
        color = np.random.rand(3, )  # Random color
        circle = plt.Circle((x, y), radius, color=color, ec="none", lw=1.5)
        ax.add_patch(circle)

    plt.show()

generate_smaller_dispersed_circles()
