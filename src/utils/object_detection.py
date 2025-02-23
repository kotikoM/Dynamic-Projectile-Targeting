import numpy as np
import cv2

from collections import deque

from src.utils.visualization import plot_dbscan_labels, plot_targets


class ImageProcessor:
    """ Class to handle image processing tasks """

    @staticmethod
    def gaussian_blur(image):
        """ Apply Gaussian blur to the input image and normalize the output. """
        gaussian_kernel = np.array([[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
                                   dtype=np.float32)
        return ImageProcessor.convolve(image, gaussian_kernel)

    @staticmethod
    def convolve(image, kernel):
        """ Apply convolution between the image and a kernel. """
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        convolved_image = np.zeros((image_height, image_width), dtype=np.float32)

        # Pad the image to handle borders
        padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')

        # Convolution operation
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                convolved_image[i, j] = np.sum(region * kernel)

        return convolved_image

    @staticmethod
    def sobel_edge_detection(image):
        """ Perform Sobel edge detection. """
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        grad_x = ImageProcessor.convolve(image, sobel_x)
        grad_y = ImageProcessor.convolve(image, sobel_y)

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)


def dbscan(points, eps=2, min_samples=5):
    """ Perform DBSCAN clustering. """
    n_points, dim = points.shape
    print(f'Clustering {n_points} data points.')
    labels = -2 * np.ones(n_points, dtype=int)  # -2 means unvisited
    cluster_id = 0

    # Function to find neighbors within 'eps' distance
    def region_query(point_idx):
        point = points[point_idx]
        return np.where(np.linalg.norm(points - point, axis=1) < eps)[0]

    # Function to expand the cluster
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()
            if labels[neighbor_idx] == -2:  # Point has not been visited
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = region_query(neighbor_idx)
                if len(neighbor_neighbors) >= min_samples:
                    queue.extend(neighbor_neighbors)
            elif labels[neighbor_idx] == -1:  # Point was marked as noise
                labels[neighbor_idx] = cluster_id

    # DBSCAN algorithm
    for point_idx in range(n_points):
        if labels[point_idx] != -2:
            continue  # Skip if already processed or marked as noise

        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1

    return labels


class TargetExtractor:
    """ Class to extract targets from images """

    def __init__(self, image_path, width, height, mode):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (width, height))
        self.mode = mode
        print(f"Loaded Image with dimensions: {self.image.shape[0]} x {self.image.shape[1]}.")

    def calculate_center_and_radius(self, data_points):
        """Calculate the center and radius of a cluster of points, adjusting y-axis behavior."""
        # Calculate the center as the mean of the x and y coordinates
        center_x = np.mean(data_points[:, 1])  # X coordinates are at index 1
        center_y = np.mean(data_points[:, 0])  # Y coordinates are at index 0

        # Calculate the radius as the average distance from the center
        distances = np.sqrt((data_points[:, 1] - center_x) ** 2 + (data_points[:, 0] - center_y) ** 2)
        radius = np.mean(distances)

        # Return center (x, y, radius)
        # Invert the y-coordinate for 1st quarter behavior
        return center_x, self.image.shape[0] - center_y, radius

    def extract(self, threshold):
        if self.mode == "visualize": cv2.imshow("Original Image", self.image)

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred_image = ImageProcessor.gaussian_blur(gray_image)
        print('Blurred input image.')

        edges = ImageProcessor.sobel_edge_detection(blurred_image)
        print('Applied Sobel edge detection.')
        if self.mode == "visualize": cv2.imshow("Edges", edges)

        points = np.argwhere(edges > threshold)

        if self.mode == "visualize":
            filtered_image = np.zeros_like(edges, dtype=np.uint8)
            for point in points:
                y, x = point
                filtered_image[y, x] = 255
            cv2.imshow("Filtered", filtered_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        labels = dbscan(points)
        print('Applied DBSCAN clustering.')
        if self.mode == "visualize": plot_dbscan_labels(points, labels)

        centers_and_radii = []
        for label in np.unique(labels):
            if label == -1:
                continue  # Skip noise points

            cluster_points = points[labels == label]
            center_x, center_y, radius = self.calculate_center_and_radius(cluster_points)
            centers_and_radii.append((center_x, center_y, radius))
        print('Calculated centers and radii of clusters.')
        if self.mode == "visualize": plot_targets(centers_and_radii)
        return centers_and_radii
