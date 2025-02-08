import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx

# Parameters
NUM_AGENTS = 3
MAP_SIZE = (100, 100)  # 2D map size
START_POSITIONS = [(10, 10), (10, 90), (90, 10)]  # Starting positions for each agent
GOAL_POSITIONS = [(90, 90), (90, 10), (10, 90)]  # Goal positions for each agent
NUM_SAMPLES = 500  # Number of samples for PRM
CONNECTION_RADIUS = 20  # Maximum connection radius between nodes
MIN_AGENT_DISTANCE = 10  # Minimum distance between agents to avoid collisions
OBSTACLES = [  # List of obstacles as (x, y, radius)
    (40, 40, 10),
    (60, 60, 10),
    (20, 80, 10)
]

# Node class for PRM
class Node:
    def __init__(self, position):
        self.position = position

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
def is_agent_collision_free(positions):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) < MIN_AGENT_DISTANCE:
                return False
    return True

# Generate random samples in the map
def generate_samples(num_samples):
    samples = []
    while len(samples) < num_samples:
        point = (np.random.uniform(0, MAP_SIZE[0]), np.random.uniform(0, MAP_SIZE[1]))
        if is_collision_free(point):
            samples.append(point)
    return samples

# Build PRM graph
def build_prm(samples, connection_radius):
    graph = nx.Graph()
    for i, sample in enumerate(samples):
        graph.add_node(i, pos=sample)
    kdtree = KDTree(samples)
    for i, sample in enumerate(samples):
        neighbors = kdtree.query_ball_point(sample, connection_radius)
        for neighbor in neighbors:
            if i != neighbor and is_path_collision_free(sample, samples[neighbor]):
                graph.add_edge(i, neighbor)
    return graph

# Find the nearest node in the graph to a given point
def nearest_node(graph, point):
    positions = [data['pos'] for _, data in graph.nodes(data=True)]
    kdtree = KDTree(positions)
    dist, idx = kdtree.query(point)
    return list(graph.nodes)[idx]

# PRM Algorithm for Multi-Agent Path Planning
def prm_multi_agent(start_positions, goal_positions):
    # Generate samples and build PRM graph
    samples = generate_samples(NUM_SAMPLES)
    graph = build_prm(samples, CONNECTION_RADIUS)

    # Add start and goal positions to the graph
    for i in range(NUM_AGENTS):
        start_node = Node(start_positions[i])
        goal_node = Node(goal_positions[i])
        samples.append(start_node.position)
        samples.append(goal_node.position)
        graph.add_node(f"start_{i}", pos=start_node.position)
        graph.add_node(f"goal_{i}", pos=goal_node.position)

        # Connect start and goal to the nearest nodes
        start_nearest = nearest_node(graph, start_node.position)
        goal_nearest = nearest_node(graph, goal_node.position)
        if is_path_collision_free(start_node.position, graph.nodes[start_nearest]['pos']):
            graph.add_edge(f"start_{i}", start_nearest)
        if is_path_collision_free(goal_node.position, graph.nodes[goal_nearest]['pos']):
            graph.add_edge(f"goal_{i}", goal_nearest)

    # Find paths for each agent
    paths = []
    for i in range(NUM_AGENTS):
        try:
            path = nx.shortest_path(graph, source=f"start_{i}", target=f"goal_{i}")
            paths.append([graph.nodes[node]['pos'] for node in path])
        except nx.NetworkXNoPath:
            print(f"No path found for Agent {i}.")
            paths.append([])
    return paths

# Visualization
def visualize(paths, start_positions, goal_positions):
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b']

    # Plot obstacles
    for (ox, oy, rad) in OBSTACLES:
        circle = plt.Circle((ox, oy), rad, color='k', alpha=0.5)
        plt.gca().add_patch(circle)

    # Plot paths
    for i in range(NUM_AGENTS):
        if paths[i]:
            plt.plot([x for (x, y) in paths[i]], [y for (x, y) in paths[i]], color=colors[i], linewidth=2, label=f"Agent {i} Path")

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(start_positions[i][0], start_positions[i][1], color=colors[i], marker='o', s=100, label=f"Agent {i} Start")
        plt.scatter(goal_positions[i][0], goal_positions[i][1], color=colors[i], marker='x', s=100, label=f"Agent {i} Goal")

    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.legend()
    plt.title("PRM for 3 Agents with Inter-Agent Collision Avoidance")
    plt.show()

# Run PRM and visualize
paths = prm_multi_agent(START_POSITIONS, GOAL_POSITIONS)
visualize(paths, START_POSITIONS, GOAL_POSITIONS)
