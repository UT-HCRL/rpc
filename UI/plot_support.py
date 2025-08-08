import matplotlib.pyplot as plt

def plot_support_polygons(*polygons_with_points):
    """
    Plots a 2D figure of multiple support polygons given their vertices and marks specific points with an "X".

    Parameters:
        polygons_with_points (list of tuple): Each tuple contains:
            - A list of (x, y) tuples representing the vertices of the polygon.
            - A tuple (x, y) representing the point where the "X" should be drawn.
    """
    if not polygons_with_points:
        raise ValueError("At least one polygon must be provided.")

    plt.figure(figsize=(6, 6))

    for i, (vertices, point) in enumerate(polygons_with_points):
        if len(vertices) < 3:
            raise ValueError(f"Polygon {i + 1} must have at least 3 vertices.")

        vertices.append(vertices[0])
        x_coords, y_coords = zip(*vertices)
        
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Support Polygon {i + 1}')
        plt.fill(x_coords, y_coords, alpha=0.5, label=f'Polygon {i + 1} Fill')

        if point:
            plt.plot(point[0], point[1], marker='x', markersize=10, label=f'Point {i + 1}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Support Polygons with Points')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    polygon1 = [(-0.0042, -0.1435), (0.1158, -0.1435), (0.3455, 0.3718), (-0.0042, -0.0935)]
    point1 = (0.110879, 0.0144731) 

    polygon2 = [(0.0035, -0.1435), (0.1235, -0.1435), (0.1235, 0.1435), (0.0035, 0.1435)]
    point2 = (0.136516, 0.000444327)

    plot_support_polygons((polygon1, point1), (polygon2, point2))