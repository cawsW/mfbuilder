from shapely.geometry import Point, Polygon
from scipy.spatial import distance
from scipy.spatial import ConvexHull
import random
import numpy as np


def midpoint(p1, p2):
    """Calculate the midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def closest_pair(points, criteria):
    """For each point, find its closest neighbor."""
    min_distances = [float('inf')] * len(points)
    pairs = [None] * len(points)

    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                d = distance.euclidean(p1, p2)
                if d < min_distances[i] and d > criteria:
                    min_distances[i] = d
                    pairs[i] = p2

    return pairs


def keep_min_distance_points(points, min_distance):
    """Filter points to maintain a minimum distance from each other."""
    kept_points = []
    for point in points:
        if all(distance.euclidean(point, kept_point) >= min_distance for kept_point in kept_points):
            kept_points.append(point)
    return kept_points


def generate_points_within_polygon(polygon, existing_points, grid_spacing=1.0, min_distance=1):
    """Generate evenly distributed points within a given polygon."""
    closest_neighbors = closest_pair(existing_points, min_distance)

    # Generate midpoints between each point and its closest neighbor
    between_points = [midpoint(p, closest_neighbors[i]) for i, p in enumerate(existing_points)]

    # Compute the bounding box
    minx, miny, maxx, maxy = polygon.bounds

    # Generate candidate points within the bounding box
    x_coords = np.arange(minx, maxx, grid_spacing)
    y_coords = np.arange(miny, maxy, grid_spacing)
    half_sp = grid_spacing / 2
    x_coords_tr = np.arange(minx + half_sp, maxx - half_sp, grid_spacing)
    y_coords_tr = np.arange(miny + half_sp, maxy - half_sp, grid_spacing)

    grid_points = []
    for x in x_coords:
        for y in y_coords:
            candidate = Point(x, y)
            if polygon.contains(candidate):
                grid_points.append((x, y))

    for x in x_coords_tr:
        for y in y_coords_tr:
            candidate = Point(x, y)
            if polygon.contains(candidate):
                grid_points.append((x, y))
    # Combine points
    all_new_points = between_points + grid_points

    # Remove duplicates and points that coincide with existing points
    all_new_points = list(set(all_new_points) - set(existing_points))
    # Filter points based on minimal distance
    final_new_points = keep_min_distance_points(all_new_points, min_distance)

    return final_new_points


def generate_random_polygon(num_points=10):
    # Create random points
    points = np.random.rand(num_points, 2) * 10  # Adjust this for the scale of your coordinates
    hull = ConvexHull(points)
    polygon_points = [tuple(point) for point in hull.points[hull.vertices]]
    return Polygon(polygon_points)


# Generate random points within a given polygon
def generate_random_points_in_polygon(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if random_point.within(polygon):
            points.append(random_point)
    return [point.coords[0] for point in points]
