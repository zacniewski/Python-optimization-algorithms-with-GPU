import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Parameters
NUM_AGENTS = 3
MAP_SIZE = (20, 20)  # 20x20 map
START_POSITIONS = [np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([9.0, 1.0])]  # Starting positions
GOAL_POSITIONS = [np.array([9.0, 9.0]), np.array([5.0, 9.0]), np.array([1.0, 9.0])]  # Goal positions
AGENT_RADIUS = 0.5  # Radius of each agent
STEP_SIZE = 0.5  # Step size for RRT*
MAX_ITER = 5000  # Maximum iterations for RRT*
REWIRE_RADIUS = 1.5  # Radius for rewiring in RRT*

# Check for collisions between two agents
def is_collision(pos1, pos2, radius):
    return np.linalg.norm(pos1 - pos2) < 2 * radius

# RRT* algorithm for a single agent
def rrt_star(start, goal, obstacles, step_size, max_iter, rewire_radius):
    tree = KDTree([start])
    nodes = [start]
    costs = [0.0]  # Cost to reach each node
    parents = [0]  # Parent of each node
    path = []

    for _ in range(max_iter):
        # Sample a random point
        random_point = np.random.uniform(low=[0, 0], high=MAP_SIZE)

        # Find the nearest node in the tree
        nearest_idx = tree.query(random_point)[1]
        nearest_node = nodes[nearest_idx]

        # Move towards the random point
        direction = normalized(random_point - nearest_node)
        new_node = nearest_node + direction * step_size

        # Check for collisions with obstacles
        collision = False
        for obstacle in obstacles:
            if is_collision(new_node, obstacle, AGENT_RADIUS):
                collision = True
                break

        if not collision:
            # Find nearby nodes within the rewiring radius
            nearby_indices = tree.query_ball_point(new_node, rewire_radius)
            nearby_nodes = [nodes[i] for i in nearby_indices]
            nearby_costs = [costs[i] for i in nearby_indices]

            # Choose the parent with the lowest cost
            min_cost = float('inf')
            best_parent_idx = nearest_idx
            for i, nearby_node in enumerate(nearby_nodes):
                cost = nearby_costs[i] + np.linalg.norm(new_node - nearby_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent_idx = nearby_indices[i]

            # Add the new node to the tree
            nodes.append(new_node)
            costs.append(min_cost)
            parents.append(best_parent_idx)
            tree = KDTree(nodes)

            # Rewire the tree
            for i, nearby_node in enumerate(nearby_nodes):
                cost = min_cost + np.linalg.norm(new_node - nearby_node)
                if cost < costs[nearby_indices[i]]:
                    costs[nearby_indices[i]] = cost
                    parents[nearby_indices[i]] = len(nodes) - 1

            # Check if the goal is reached
            if np.linalg.norm(new_node - goal) < step_size:
                path = []
                current_idx = len(nodes) - 1
                while current_idx != 0:
                    path.append(nodes[current_idx])
                    current_idx = parents[current_idx]
                path.append(nodes[0])
                return path[::-1]  # Reverse the path

    return None  # No path found

# Normalize a vector
def normalized(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

# Multi-Agent RRT*
def multi_agent_rrt_star(starts, goals, num_agents, step_size, max_iter, rewire_radius):
    paths = []
    obstacles = []

    for i in range(num_agents):
        # Plan path for the current agent
        path = rrt_star(starts[i], goals[i], obstacles, step_size, max_iter, rewire_radius)
        if path is None:
            print(f"No path found for Agent {i+1}")
            return None
        paths.append(path)

        # Add the planned path as obstacles for the next agents
        for point in path:
            obstacles.append(point)

    return paths

# Run Multi-Agent RRT*
paths = multi_agent_rrt_star(START_POSITIONS, GOAL_POSITIONS, NUM_AGENTS, STEP_SIZE, MAX_ITER, REWIRE_RADIUS)

# Plot the results
if paths is not None:
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b']
    for i, path in enumerate(paths):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color=colors[i], label=f"Agent {i+1}")
        plt.scatter(START_POSITIONS[i][0], START_POSITIONS[i][1], marker='o', color=colors[i], label=f"Start {i+1}")
        plt.scatter(GOAL_POSITIONS[i][0], GOAL_POSITIONS[i][1], marker='x', color=colors[i], label=f"Goal {i+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Multi-Agent Path Planning with RRT*")
    plt.legend()
    plt.grid()
    plt.show()
