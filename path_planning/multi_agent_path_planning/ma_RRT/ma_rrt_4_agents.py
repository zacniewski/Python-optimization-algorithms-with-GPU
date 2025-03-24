import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Parameters - Updated for 4 agents and 4 obstacles
NUM_AGENTS = 4
MAP_SIZE = (100, 100)  # 2D map size
START_POSITIONS = [(10, 10), (10, 90), (90, 10), (90, 90)]  # Starting positions for each agent
GOAL_POSITIONS = [(90, 90), (90, 10), (10, 90), (10, 10)]  # Goal positions for each agent
MAX_ITER = 1000  # Maximum iterations
STEP_SIZE = 5  # Maximum step size for each agent
GOAL_RADIUS = 5  # Radius to consider the goal reached
MIN_AGENT_DISTANCE = 10  # Minimum distance between agents to avoid collisions
OBSTACLES = [  # List of obstacles as (x, y, radius)
    (40, 40, 10),
    (60, 60, 10),
    (20, 80, 10),
    (80, 20, 10)  # New obstacle added
]

# Node class for RRT
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

# Check if a point is in collision with obstacles
def is_collision_free(point):
    for (ox, oy, rad) in OBSTACLES:
        if np.linalg.norm(np.array(point) - np.array((ox, oy))) < rad:
            return False
    return True

# Check if a path between two points is collision-free
def is_path_collision_free(start, end):
    points = np.linspace(start, end, num=10)
    for point in points:
        if not is_collision_free(point):
            return False
    return True

# Check if agents are too close to each other
def is_agent_collision_free(new_positions):
    for i in range(len(new_positions)):
        for j in range(i + 1, len(new_positions)):
            if np.linalg.norm(np.array(new_positions[i]) - np.array(new_positions[j])) < MIN_AGENT_DISTANCE:
                return False
    return True

# Generate a random point in the map
def random_point():
    return (np.random.uniform(0, MAP_SIZE[0]), np.random.uniform(0, MAP_SIZE[1]))

# Find the nearest node in the tree to a given point
def nearest_node(tree, point):
    positions = [node.position for node in tree]
    kdtree = KDTree(positions)
    dist, idx = kdtree.query(point)
    return tree[idx]

# Steer from one point toward another
def steer(from_point, to_point, step_size):
    direction = np.array(to_point) - np.array(from_point)
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return to_point
    else:
        return tuple(np.array(from_point) + (direction / distance) * step_size)

# MA-RRT Algorithm
def ma_rrt(start_positions, goal_positions):
    trees = [[] for _ in range(NUM_AGENTS)]  # Separate trees for each agent
    for i in range(NUM_AGENTS):
        trees[i].append(Node(start_positions[i]))  # Initialize each tree with the start position

    for _ in range(MAX_ITER):
        new_positions = [None] * NUM_AGENTS
        for i in range(NUM_AGENTS):
            random_point_i = random_point()
            nearest_node_i = nearest_node(trees[i], random_point_i)
            new_point_i = steer(nearest_node_i.position, random_point_i, STEP_SIZE)

            if is_path_collision_free(nearest_node_i.position, new_point_i):
                new_positions[i] = new_point_i

        # Check inter-agent collisions
        if new_positions.count(None) == 0 and is_agent_collision_free(new_positions):
            for i in range(NUM_AGENTS):
                new_node_i = Node(new_positions[i], nearest_node(trees[i], new_positions[i]))
                trees[i].append(new_node_i)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_positions[i]) - np.array(goal_positions[i])) < GOAL_RADIUS:
                    print(f"Agent {i} reached the goal!")
                    return trees

    print("Maximum iterations reached. Some agents may not have reached the goal.")
    return trees

# Reconstruct the path from the tree
def reconstruct_path(tree, goal):
    path = []
    node = nearest_node(tree, goal)
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

# Visualization
def visualize(trees, start_positions, goal_positions):
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'm']  # Added magenta for the 4th agent

    # Plot obstacles
    for (ox, oy, rad) in OBSTACLES:
        circle = plt.Circle((ox, oy), rad, color='k', alpha=0.5)
        plt.gca().add_patch(circle)

    # Plot trees and paths
    for i in range(NUM_AGENTS):
        for node in trees[i]:
            if node.parent:
                plt.plot([node.position[0], node.parent.position[0]],
                         [node.position[1], node.parent.position[1]],
                         color=colors[i], alpha=0.3)

        # Reconstruct and plot the path
        path = reconstruct_path(trees[i], goal_positions[i])
        plt.plot([x for (x, y) in path], [y for (x, y) in path], color=colors[i], linewidth=2, label=f"Agent {i} Path")

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(start_positions[i][0], start_positions[i][1], color=colors[i], marker='o', s=100, label=f"Agent {i} Start")
        plt.scatter(goal_positions[i][0], goal_positions[i][1], color=colors[i], marker='x', s=100, label=f"Agent {i} Goal")

    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.legend()
    plt.title("MA-RRT for 4 Agents with Inter-Agent Collision Avoidance")
    plt.show()

# Run MA-RRT and visualize
trees = ma_rrt(START_POSITIONS, GOAL_POSITIONS)
visualize(trees, START_POSITIONS, GOAL_POSITIONS)
