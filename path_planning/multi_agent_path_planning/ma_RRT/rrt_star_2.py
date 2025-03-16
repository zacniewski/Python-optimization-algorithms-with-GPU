import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Parameters
NUM_AGENTS = 3
MAP_SIZE = (10, 10)  # 10x10 map
START_POSITIONS = [np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([9.0, 1.0])]  # Starting positions
GOAL_POSITIONS = [np.array([9.0, 9.0]), np.array([5.0, 9.0]), np.array([1.0, 9.0])]  # Goal positions
AGENT_RADIUS = 0.5  # Radius of each agent
STEP_SIZE = 0.5  # Step size for RRT*
MAX_ITER = 5000  # Maximum iterations for RRT*
REWIRE_RADIUS = 1.5  # Radius for rewiring in RRT*

# Define obstacles as rectangles (x1, y1, x2, y2)
OBSTACLES = [
    (2.0, 3.0, 4.0, 5.0),  # Obstacle 1
    (6.0, 2.0, 8.0, 4.0),  # Obstacle 2
    (3.0, 6.0, 7.0, 8.0)  # Obstacle 3
]


# Check for collisions between two agents
def is_collision(pos1, pos2, radius):
    return np.linalg.norm(pos1 - pos2) < 2 * radius


# Check if a point is inside an obstacle
def is_point_in_obstacle(point, obstacles):
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            return True
    return False


# Check if a line segment intersects an obstacle
def is_line_in_obstacle(start, end, obstacles):
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle
        # Check if the line segment intersects the rectangle
        if (start[0] < x1 and end[0] < x1) or (start[0] > x2 and end[0] > x2):
            continue  # No intersection in x-axis
        if (start[1] < y1 and end[1] < y1) or (start[1] > y2 and end[1] > y2):
            continue  # No intersection in y-axis
        return True  # Intersection detected
    return False


# RRT* algorithm for a single agent
def rrt_star(start, goal, obstacles, other_agents, step_size, max_iter, rewire_radius):
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

        # Check if the new node is inside an obstacle
        if is_point_in_obstacle(new_node, obstacles):
            continue  # Skip this point if it's inside an obstacle

        # Check if the line segment from nearest_node to new_node intersects an obstacle
        if is_line_in_obstacle(nearest_node, new_node, obstacles):
            continue  # Skip this point if the path intersects an obstacle

        # Check for collisions with other agents
        collision_with_agent = False
        for agent in other_agents:
            if is_collision(new_node, agent, AGENT_RADIUS):
                collision_with_agent = True
                break
        if collision_with_agent:
            continue  # Skip this point if it collides with another agent

        # Find nearby nodes within the rewiring radius
        nearby_indices = tree.query_ball_point(new_node, rewire_radius)
        nearby_nodes = [nodes[i] for i in nearby_indices]
        nearby_costs = [costs[i] for i in nearby_indices]

        # Choose the parent with the lowest cost
        min_cost = float('inf')
        best_parent_idx = nearest_idx
        for i, nearby_node in enumerate(nearby_nodes):
            # Check if the line segment from nearby_node to new_node is collision-free
            if not is_line_in_obstacle(nearby_node, new_node, obstacles):
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
            # Check if the line segment from new_node to nearby_node is collision-free
            if not is_line_in_obstacle(new_node, nearby_node, obstacles):
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
def multi_agent_rrt_star(starts, goals, num_agents, step_size, max_iter, rewire_radius, obstacles):
    paths = []
    agent_positions = starts.copy()

    for i in range(num_agents):
        # Plan path for the current agent
        path = rrt_star(starts[i], goals[i], obstacles, agent_positions[:i] + agent_positions[i + 1:], step_size,
                        max_iter, rewire_radius)
        if path is None:
            print(f"No path found for Agent {i + 1}")
            return None
        paths.append(path)

        # Update the agent's position to the goal (for collision avoidance with other agents)
        agent_positions[i] = goals[i]

    return paths


# Run Multi-Agent RRT*
paths = multi_agent_rrt_star(START_POSITIONS, GOAL_POSITIONS, NUM_AGENTS, STEP_SIZE, MAX_ITER, REWIRE_RADIUS, OBSTACLES)

# Plot the results
if paths is not None:
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b']
    for i, path in enumerate(paths):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color=colors[i], label=f"Agent {i + 1}")
        plt.scatter(START_POSITIONS[i][0], START_POSITIONS[i][1], marker='o', color=colors[i], label=f"Start {i + 1}")
        plt.scatter(GOAL_POSITIONS[i][0], GOAL_POSITIONS[i][1], marker='x', color=colors[i], label=f"Goal {i + 1}")

    # Plot obstacles
    for obstacle in OBSTACLES:
        x1, y1, x2, y2 = obstacle
        plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color='gray', alpha=0.5, label="Obstacle")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Multi-Agent Path Planning with RRT* and Obstacles")
    plt.legend()
    plt.grid()
    plt.show()
