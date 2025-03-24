import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# Environment Parameters
NUM_AGENTS = 4
MAP_SIZE = (100, 100)
START_POSITIONS = [(10, 10), (10, 90), (90, 10), (90, 90)]
GOAL_POSITIONS = [(90, 90), (90, 10), (10, 90), (10, 10)]
OBSTACLES = [(40, 40, 5), (60, 60, 5), (20, 80, 5), (80, 20, 5)]

# Algorithm Parameters
MAX_ITER = 5000
STEP_SIZE = 5
GOAL_RADIUS = 3
MIN_AGENT_DISTANCE = 4
INITIAL_GOAL_BIAS = 0.2
MAX_GOAL_BIAS = 0.5

# Visualization
COLORS = ['r', 'g', 'b', 'm']
VISUALIZE_EVERY = 25
PAUSE_TIME = 0.05


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
    return all(is_collision_free(p) for p in points)


def is_agent_collision_free(positions):  # Corrected spelling here
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) < MIN_AGENT_DISTANCE:
                return False
    return True


def random_point(i, bias):
    if np.random.random() < bias:
        return GOAL_POSITIONS[i]
    return (np.random.uniform(0, MAP_SIZE[0]), np.random.uniform(0, MAP_SIZE[1]))


def nearest_node(tree, point):
    positions = [node.position for node in tree]
    kdtree = KDTree(positions)
    _, idx = kdtree.query(point)
    return tree[idx]


def steer(from_point, to_point, step_size):
    direction = np.array(to_point) - np.array(from_point)
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return to_point
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
    for (ox, oy, rad) in OBSTACLES:
        plt.gca().add_patch(plt.Circle((ox, oy), rad, color='k', alpha=0.5))
    for i in range(NUM_AGENTS):
        plt.scatter(*START_POSITIONS[i], color=COLORS[i], marker='o', s=100, label=f'Agent {i} Start')
        plt.scatter(*GOAL_POSITIONS[i], color=COLORS[i], marker='x', s=100, label=f'Agent {i} Goal')
    plt.legend()
    plt.title("MA-RRT Initial Setup")
    plt.draw()
    plt.pause(1)


def update_visualization(trees, iteration, goals_reached):
    plt.clf()
    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.title(f"Iteration: {iteration} | Goals Reached: {sum(goals_reached)}/{NUM_AGENTS}")

    for (ox, oy, rad) in OBSTACLES:
        plt.gca().add_patch(plt.Circle((ox, oy), rad, color='k', alpha=0.5))

    for i in range(NUM_AGENTS):
        for node in trees[i]:
            if node.parent:
                plt.plot([node.position[0], node.parent.position[0]],
                         [node.position[1], node.parent.position[1]],
                         color=COLORS[i], alpha=0.2, linewidth=0.8)

        path = reconstruct_path(trees[i], GOAL_POSITIONS[i])
        if len(path) > 1:
            plt.plot(*zip(*path), color=COLORS[i], linewidth=2,
                     linestyle='--' if goals_reached[i] else '-')

        if not goals_reached[i]:
            last_pos = trees[i][-1].position
            plt.scatter(*last_pos, color=COLORS[i], s=50, edgecolors='k')
            plt.text(last_pos[0], last_pos[1], str(i), color='white',
                     ha='center', va='center', fontsize=8,
                     bbox=dict(facecolor=COLORS[i], alpha=0.7, boxstyle='circle'))

    for i in range(NUM_AGENTS):
        plt.scatter(*START_POSITIONS[i], color=COLORS[i], marker='o', s=50)
        plt.scatter(*GOAL_POSITIONS[i], color=COLORS[i], marker='x', s=100)

    plt.draw()
    plt.pause(PAUSE_TIME)


def ma_rrt_parallel_optimized():
    trees = [[Node(start)] for start in START_POSITIONS]
    goals_reached = [False] * NUM_AGENTS
    init_visualization()

    for iteration in range(MAX_ITER):
        current_bias = min(INITIAL_GOAL_BIAS * (1 + iteration / 300), MAX_GOAL_BIAS)
        agent_order = np.random.permutation(NUM_AGENTS)
        new_positions = [None] * NUM_AGENTS

        for i in agent_order:
            if goals_reached[i]:
                continue

            rand_point = random_point(i, current_bias)
            nearest = nearest_node(trees[i], rand_point)
            new_point = steer(nearest.position, rand_point, STEP_SIZE)

            if is_path_collision_free(nearest.position, new_point):
                temp_positions = new_positions.copy()
                temp_positions[i] = new_point

                all_positions = []
                for j in range(NUM_AGENTS):
                    if temp_positions[j] is not None:
                        all_positions.append(temp_positions[j])
                    elif goals_reached[j]:
                        all_positions.append(GOAL_POSITIONS[j])
                    else:
                        all_positions.append(trees[j][-1].position)

                if is_agent_collision_free(all_positions):  # Now using correctly spelled function
                    new_positions[i] = new_point

        for i in range(NUM_AGENTS):
            if new_positions[i] is not None:
                trees[i].append(Node(new_positions[i], nearest_node(trees[i], new_positions[i])))

                if np.linalg.norm(np.array(new_positions[i]) - np.array(GOAL_POSITIONS[i])) < GOAL_RADIUS:
                    goals_reached[i] = True
                    print(f"Agent {i} reached goal at iteration {iteration}")

        if iteration % VISUALIZE_EVERY == 0 or all(goals_reached):
            update_visualization(trees, iteration, goals_reached)
            if all(goals_reached):
                break

    print("\n=== Final Results ===")
    for i in range(NUM_AGENTS):
        status = "SUCCESS" if goals_reached[i] else "FAILED"
        print(f"Agent {i}: {status} (Path length: {len(trees[i])})")

    plt.title(f"MA-RRT Final Result\n{sum(goals_reached)}/{NUM_AGENTS} agents succeeded")
    plt.show()
    return trees


# Run the optimized algorithm
final_trees = ma_rrt_parallel_optimized()
