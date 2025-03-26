import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import matplotlib.animation as animation

# Parameters
NUM_AGENTS = 3
MAP_SIZE = (10, 10)
START_POSITIONS = [np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([9.0, 1.0])]
GOAL_POSITIONS = [np.array([9.0, 9.0]), np.array([5.0, 9.0]), np.array([1.0, 9.0])]
AGENT_RADIUS = 0.5
STEP_SIZE = 0.5
MAX_ITER = 1000
REWIRE_RADIUS = 1.5
OBSTACLES = [(2.0, 3.0, 4.0, 5.0), (6.0, 2.0, 8.0, 4.0), (3.0, 6.0, 7.0, 8.0)]


def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def is_point_in_obstacle(point, obstacles):
    point = np.array(point)
    for (x1, y1, x2, y2) in obstacles:
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            return True
    return False


def line_segments_intersect(a1, a2, b1, b2):
    """Check if line segments a1-a2 and b1-b2 intersect"""
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    # Vector calculations
    v1 = a2 - a1
    v2 = b2 - b1
    v3 = b1 - a1

    # Cross products
    cross_v1_v2 = np.cross(v1, v2)

    # Handle parallel lines case
    if np.abs(cross_v1_v2) < 1e-6:  # Using np.abs for array
        return False

    t = np.cross(v3, v2) / cross_v1_v2
    u = np.cross(v3, v1) / cross_v1_v2

    return (0 <= t <= 1) and (0 <= u <= 1)


def line_rect_intersect(p1, p2, rect):
    """Check if line segment p1-p2 intersects rectangle rect=(x1,y1,x2,y2)"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    x1, y1, x2, y2 = rect

    if is_point_in_obstacle(p1, [rect]) or is_point_in_obstacle(p2, [rect]):
        return True

    edges = [
        (np.array([x1, y1]), np.array([x2, y1])),  # Bottom
        (np.array([x2, y1]), np.array([x2, y2])),  # Right
        (np.array([x2, y2]), np.array([x1, y2])),  # Top
        (np.array([x1, y2]), np.array([x1, y1]))  # Left
    ]

    for i in range(4):
        edge_start = edges[i][0]
        edge_end = edges[i][1]
        if line_segments_intersect(p1, p2, edge_start, edge_end):
            return True
    return False


def is_line_in_obstacle(start, end, obstacles):
    for obstacle in obstacles:
        if line_rect_intersect(start, end, obstacle):
            return True
    return False


def rrt_star(start, goal, obstacles, step_size, max_iter, rewire_radius):
    tree = KDTree([start])
    nodes = [start]
    costs = [0.0]
    parents = [0]
    path = []

    for _ in range(max_iter):
        if np.random.random() < 0.1:
            random_point = goal
        else:
            random_point = np.random.uniform(low=[0, 0], high=MAP_SIZE)

        nearest_idx = tree.query(random_point)[1]
        nearest_node = nodes[nearest_idx]

        direction = normalized(random_point - nearest_node)
        new_node = nearest_node + direction * step_size

        if (is_point_in_obstacle(new_node, obstacles) or
                is_line_in_obstacle(nearest_node, new_node, obstacles)):
            continue

        nearby_indices = tree.query_ball_point(new_node, rewire_radius)
        nearby_nodes = [nodes[i] for i in nearby_indices]
        nearby_costs = [costs[i] for i in nearby_indices]

        min_cost = float('inf')
        best_parent_idx = nearest_idx
        for i, node in enumerate(nearby_nodes):
            if not is_line_in_obstacle(node, new_node, obstacles):
                cost = nearby_costs[i] + np.linalg.norm(new_node - node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent_idx = nearby_indices[i]

        nodes.append(new_node)
        costs.append(min_cost)
        parents.append(best_parent_idx)
        tree = KDTree(nodes)

        for i, node in enumerate(nearby_nodes):
            if not is_line_in_obstacle(new_node, node, obstacles):
                new_cost = min_cost + np.linalg.norm(new_node - node)
                if new_cost < costs[nearby_indices[i]]:
                    costs[nearby_indices[i]] = new_cost
                    parents[nearby_indices[i]] = len(nodes) - 1

        if np.linalg.norm(new_node - goal) < step_size:
            path = []
            current_idx = len(nodes) - 1
            while current_idx != 0:
                path.append(nodes[current_idx])
                current_idx = parents[current_idx]
            path.append(nodes[0])
            return path[::-1]
    return None


def run_visualization():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])
    ax.set_title("Multi-Agent RRT* Path Planning")
    colors = ['red', 'green', 'blue']

    for (x1, y1, x2, y2) in OBSTACLES:
        ax.fill([x1, x2, x2, x1], [y1, y1, y2, y2], 'gray', alpha=0.5)

    agents = []
    for i in range(NUM_AGENTS):
        ax.scatter(*START_POSITIONS[i], color=colors[i], marker='o', s=100)
        ax.scatter(*GOAL_POSITIONS[i], color=colors[i], marker='x', s=100)

        tree_line, = ax.plot([], [], '.-', color=colors[i], alpha=0.3, lw=0.5)
        path_line, = ax.plot([], [], '-', color=colors[i], lw=2)
        agents.append({
            'tree': KDTree([START_POSITIONS[i]]),
            'nodes': [START_POSITIONS[i]],
            'costs': [0.0],
            'parents': [0],
            'path': None,
            'tree_line': tree_line,
            'path_line': path_line
        })

    def update(frame):
        for i, agent in enumerate(agents):
            if agent['path'] is not None:
                continue

            # RRT* iteration
            random_point = np.random.uniform(low=[0, 0], high=MAP_SIZE) if np.random.random() > 0.1 else GOAL_POSITIONS[
                i]
            nearest_idx = agent['tree'].query(random_point)[1]
            nearest_node = agent['nodes'][nearest_idx]

            direction = normalized(random_point - nearest_node)
            new_node = nearest_node + direction * STEP_SIZE

            if (not is_point_in_obstacle(new_node, OBSTACLES) and
                    not is_line_in_obstacle(nearest_node, new_node, OBSTACLES)):

                # Add new node to tree
                agent['nodes'].append(new_node)
                new_node_idx = len(agent['nodes']) - 1
                agent['parents'].append(nearest_idx)  # Track parent of new node
                agent['tree'] = KDTree(agent['nodes'])

                # Update tree visualization
                x_data = [agent['nodes'][j][0] for j in range(len(agent['nodes']))]
                y_data = [agent['nodes'][j][1] for j in range(len(agent['nodes']))]
                agent['tree_line'].set_data(x_data, y_data)

                # Check if goal reached
                if np.linalg.norm(new_node - GOAL_POSITIONS[i]) < STEP_SIZE:
                    path = []
                    current_idx = new_node_idx
                    while current_idx != 0:
                        path.append(agent['nodes'][current_idx])
                        current_idx = agent['parents'][current_idx]
                    path.append(agent['nodes'][0])  # Add start node
                    agent['path'] = path[::-1]  # Reverse to get start-to-goal path
                    path_array = np.array(agent['path'])
                    agent['path_line'].set_data(path_array[:, 0], path_array[:, 1])

        return [agent['tree_line'] for agent in agents] + [agent['path_line'] for agent in agents]

    ani = animation.FuncAnimation(fig, update, frames=MAX_ITER, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    ani = run_visualization()
    ani.save('multi_agent_rrt_star.mp4', writer='ffmpeg', fps=30)
