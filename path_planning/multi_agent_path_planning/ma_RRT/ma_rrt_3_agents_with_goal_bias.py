import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# Parameters
NUM_AGENTS = 4
MAP_SIZE = (100, 100)
START_POSITIONS = [(10, 10), (10, 90), (90, 10), (90, 90)]
GOAL_POSITIONS = [(90, 90), (90, 10), (10, 90), (10, 10)]
MAX_ITER = 5000
STEP_SIZE = 5
GOAL_RADIUS = 5
MIN_AGENT_DISTANCE = 6  # Reduced from 10 for less strict collision avoidance
GOAL_BIAS = 0.1  # 10% chance to sample goal directly

OBSTACLES = [
    (40, 40, 10),
    (60, 60, 10),
    (20, 80, 10),
    (80, 20, 10)
]

# Visualization settings
VISUALIZE_EVERY = 20
PAUSE_TIME = 0.05
COLORS = ['r', 'g', 'b', 'm']  # Colors for each agent


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent


def is_collision_free(point):
    for (ox, oy, rad) in OBSTACLES:
        if np.linalg.norm(np.array(point) - np.array((ox, oy))) < rad:
            return False
    return True


def is_path_collision_free(start, end):
    points = np.linspace(start, end, num=10)
    for point in points:
        if not is_collision_free(point):
            return False
    return True


def is_agent_collision_free(new_positions):
    for i in range(len(new_positions)):
        for j in range(i + 1, len(new_positions)):
            if np.linalg.norm(np.array(new_positions[i]) - np.array(new_positions[j])) < MIN_AGENT_DISTANCE:
                return False
    return True


def random_point(i):
    # Goal-biased sampling
    if np.random.random() < GOAL_BIAS:
        return GOAL_POSITIONS[i]
    return (np.random.uniform(0, MAP_SIZE[0]), np.random.uniform(0, MAP_SIZE[1]))


def nearest_node(tree, point):
    positions = [node.position for node in tree]
    kdtree = KDTree(positions)
    dist, idx = kdtree.query(point)
    return tree[idx]


def steer(from_point, to_point, step_size):
    direction = np.array(to_point) - np.array(from_point)
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return to_point
    else:
        return tuple(np.array(from_point) + (direction / distance) * step_size)


def reconstruct_path(tree, goal):
    path = []
    node = nearest_node(tree, goal)
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]


def init_visualization():
    plt.figure(figsize=(10, 10))
    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])

    # Plot obstacles
    for (ox, oy, rad) in OBSTACLES:
        circle = plt.Circle((ox, oy), rad, color='k', alpha=0.5)
        plt.gca().add_patch(circle)

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(START_POSITIONS[i][0], START_POSITIONS[i][1],
                    color=COLORS[i], marker='o', s=100, label=f'Agent {i} Start')
        plt.scatter(GOAL_POSITIONS[i][0], GOAL_POSITIONS[i][1],
                    color=COLORS[i], marker='x', s=100, label=f'Agent {i} Goal')

    plt.legend()
    plt.title("MA-RRT Path Planning - Initial Setup")
    plt.draw()
    plt.pause(1)  # Pause to show initial setup


def update_visualization(trees, iteration, goals_reached):
    plt.clf()
    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.title(f"MA-RRT Progress (Iteration: {iteration})\nGoals Reached: {sum(goals_reached)}/{NUM_AGENTS}")

    # Plot obstacles
    for (ox, oy, rad) in OBSTACLES:
        circle = plt.Circle((ox, oy), rad, color='k', alpha=0.5)
        plt.gca().add_patch(circle)

    # Plot trees and paths
    for i in range(NUM_AGENTS):
        # Draw the tree
        for node in trees[i]:
            if node.parent:
                plt.plot([node.position[0], node.parent.position[0]],
                         [node.position[1], node.parent.position[1]],
                         color=COLORS[i], alpha=0.2, linewidth=0.5)

        # Draw current best path
        if not goals_reached[i] or iteration % 50 == 0:  # Update path periodically
            path = reconstruct_path(trees[i], GOAL_POSITIONS[i])
            if len(path) > 1:
                plt.plot([x for (x, y) in path], [y for (x, y) in path],
                         color=COLORS[i], linewidth=2, linestyle='--' if goals_reached[i] else '-')

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(START_POSITIONS[i][0], START_POSITIONS[i][1],
                    color=COLORS[i], marker='o', s=100)
        plt.scatter(GOAL_POSITIONS[i][0], GOAL_POSITIONS[i][1],
                    color=COLORS[i], marker='x', s=100)

    plt.draw()
    plt.pause(PAUSE_TIME)


def ma_rrt_optimized():
    trees = [[] for _ in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        trees[i].append(Node(START_POSITIONS[i]))

    goals_reached = [False] * NUM_AGENTS
    init_visualization()

    for iteration in range(MAX_ITER):
        new_positions = [None] * NUM_AGENTS

        # Plan moves for all active agents
        for i in range(NUM_AGENTS):
            if goals_reached[i]:
                continue

            rand_point = random_point(i)
            nearest = nearest_node(trees[i], rand_point)
            new_point = steer(nearest.position, rand_point, STEP_SIZE)

            if is_path_collision_free(nearest.position, new_point):
                new_positions[i] = new_point

        # Execute moves if collision-free
        if None not in new_positions and is_agent_collision_free(new_positions):
            for i in range(NUM_AGENTS):
                if goals_reached[i]:
                    continue

                trees[i].append(Node(new_positions[i], nearest_node(trees[i], new_positions[i])))

                # Check goal condition
                if np.linalg.norm(np.array(new_positions[i]) - np.array(GOAL_POSITIONS[i])) < GOAL_RADIUS:
                    goals_reached[i] = True
                    print(f"Agent {i} reached goal at iteration {iteration}")

        # Visual update
        if iteration % VISUALIZE_EVERY == 0 or all(goals_reached):
            update_visualization(trees, iteration, goals_reached)
            if all(goals_reached):
                break

    # Final visualization
    update_visualization(trees, iteration, goals_reached)
    plt.title("MA-RRT Final Paths\nAll goals reached!" if all(goals_reached)
              else f"MA-RRT Final Paths\nOnly {sum(goals_reached)}/{NUM_AGENTS} goals reached")
    plt.show()

    return trees


# Run the optimized algorithm
final_trees = ma_rrt_optimized()
