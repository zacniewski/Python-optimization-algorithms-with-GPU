import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# Environment Parameters
NUM_AGENTS = 4
MAP_SIZE = (100, 100)
START_POSITIONS = [(10, 10), (10, 90), (90, 10), (90, 90)]
GOAL_POSITIONS = [(90, 90), (90, 10), (10, 90), (10, 10)]

# Random Obstacle Generation
NUM_OBSTACLES = 7
MIN_OBSTACLE_RADIUS = 3
MAX_OBSTACLE_RADIUS = 8
OBSTACLE_BUFFER = 15  # Minimum distance from start/goal positions


def generate_obstacles():
    obstacles = []
    all_points = START_POSITIONS + GOAL_POSITIONS

    for _ in range(NUM_OBSTACLES):
        while True:
            x = np.random.uniform(MAX_OBSTACLE_RADIUS, MAP_SIZE[0] - MAX_OBSTACLE_RADIUS)
            y = np.random.uniform(MAX_OBSTACLE_RADIUS, MAP_SIZE[1] - MAX_OBSTACLE_RADIUS)
            r = np.random.uniform(MIN_OBSTACLE_RADIUS, MAX_OBSTACLE_RADIUS)

            # Check distance to start/goal positions
            valid = True
            for (px, py) in all_points:
                if np.linalg.norm(np.array([x, y]) - np.array([px, py])) < OBSTACLE_BUFFER + r:
                    valid = False
                    break

            # Check distance to other obstacles
            if valid:
                for (ox, oy, orad) in obstacles:
                    if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < r + orad + 5:
                        valid = False
                        break

            if valid:
                obstacles.append((x, y, r))
                break

    return obstacles


OBSTACLES = generate_obstacles()

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


def is_agent_collision_free(positions):
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

    # Plot obstacles with random colors
    for idx, (ox, oy, rad) in enumerate(OBSTACLES):
        color = plt.cm.tab20(idx % 20)  # Cycle through colormap
        plt.gca().add_patch(plt.Circle((ox, oy), rad, color=color, alpha=0.7))
        plt.text(ox, oy, f"{rad:.1f}", ha='center', va='center', color='white')

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(*START_POSITIONS[i], color=COLORS[i], marker='o', s=100,
                    label=f'Agent {i} Start', edgecolors='k')
        plt.scatter(*GOAL_POSITIONS[i], color=COLORS[i], marker='X', s=100,
                    label=f'Agent {i} Goal', edgecolors='k')

    plt.legend()
    plt.title("MA-RRT with Random Obstacles - Initial Setup")
    plt.draw()
    plt.pause(1)


def update_visualization(trees, iteration, goals_reached):
    plt.clf()
    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.title(f"Iteration: {iteration} | Goals Reached: {sum(goals_reached)}/{NUM_AGENTS}")

    # Re-draw obstacles
    for idx, (ox, oy, rad) in enumerate(OBSTACLES):
        color = plt.cm.tab20(idx % 20)
        plt.gca().add_patch(plt.Circle((ox, oy), rad, color=color, alpha=0.3))

    # Draw trees and paths
    for i in range(NUM_AGENTS):
        # Tree branches
        for node in trees[i]:
            if node.parent:
                plt.plot([node.position[0], node.parent.position[0]],
                         [node.position[1], node.parent.position[1]],
                         color=COLORS[i], alpha=0.15, linewidth=0.7)

        # Current best path
        path = reconstruct_path(trees[i], GOAL_POSITIONS[i])
        if len(path) > 1:
            plt.plot(*zip(*path), color=COLORS[i], linewidth=2.5,
                     linestyle='--' if goals_reached[i] else '-')

        # Current position indicator
        if not goals_reached[i]:
            last_pos = trees[i][-1].position
            plt.scatter(*last_pos, color=COLORS[i], s=80, edgecolors='k', zorder=10)
            plt.text(last_pos[0], last_pos[1], str(i), color='white',
                     ha='center', va='center', fontsize=9, weight='bold',
                     bbox=dict(facecolor=COLORS[i], alpha=0.9, boxstyle='circle'))

    # Re-draw start/goal markers
    for i in range(NUM_AGENTS):
        plt.scatter(*START_POSITIONS[i], color=COLORS[i], marker='o', s=80, edgecolors='k')
        plt.scatter(*GOAL_POSITIONS[i], color=COLORS[i], marker='X', s=100, edgecolors='k')

    plt.draw()
    plt.pause(PAUSE_TIME)


def ma_rrt_with_random_obstacles():
    trees = [[Node(start)] for start in START_POSITIONS]
    goals_reached = [False] * NUM_AGENTS

    print("Generated Obstacles:")
    for i, (x, y, r) in enumerate(OBSTACLES):
        print(f"Obstacle {i}: Position=({x:.1f}, {y:.1f}), Radius={r:.1f}")

    init_visualization()

    for iteration in range(MAX_ITER):
        current_bias = min(INITIAL_GOAL_BIAS * (1 + iteration / 300), MAX_GOAL_BIAS)
        agent_order = np.random.permutation(NUM_AGENTS)
        new_positions = [None] * NUM_AGENTS

        # Planning phase
        for i in agent_order:
            if goals_reached[i]:
                continue

            rand_point = random_point(i, current_bias)
            nearest = nearest_node(trees[i], rand_point)
            new_point = steer(nearest.position, rand_point, STEP_SIZE)

            if is_path_collision_free(nearest.position, new_point):
                # Create temporary scenario
                temp_positions = new_positions.copy()
                temp_positions[i] = new_point

                # Complete the positions array
                all_positions = []
                for j in range(NUM_AGENTS):
                    if temp_positions[j] is not None:
                        all_positions.append(temp_positions[j])
                    elif goals_reached[j]:
                        all_positions.append(GOAL_POSITIONS[j])
                    else:
                        all_positions.append(trees[j][-1].position)

                if is_agent_collision_free(all_positions):
                    new_positions[i] = new_point

        # Movement phase
        for i in range(NUM_AGENTS):
            if new_positions[i] is not None:
                trees[i].append(Node(new_positions[i], nearest_node(trees[i], new_positions[i])))

                if np.linalg.norm(np.array(new_positions[i]) - np.array(GOAL_POSITIONS[i])) < GOAL_RADIUS:
                    goals_reached[i] = True
                    print(f"Agent {i} reached goal at iteration {iteration}")

        # Visualization and termination
        if iteration % VISUALIZE_EVERY == 0 or all(goals_reached):
            update_visualization(trees, iteration, goals_reached)
            if all(goals_reached):
                break

    # Final results
    print("\n=== Final Results ===")
    for i in range(NUM_AGENTS):
        path_length = len(reconstruct_path(trees[i], GOAL_POSITIONS[i]))
        status = "SUCCESS" if goals_reached[i] else "FAILED"
        print(f"Agent {i}: {status} (Path length: {path_length})")

    plt.title(f"MA-RRT Final Result\n{sum(goals_reached)}/{NUM_AGENTS} agents succeeded")
    plt.show()
    return trees


# Run the algorithm with random obstacles
final_trees = ma_rrt_with_random_obstacles()
