import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_AGENTS = 3
MAP_SIZE = (100, 100)  # 2D map size
START_POSITIONS = np.array([[10.0, 10.0], [10.0, 90.0], [90.0, 10.0]])  # Starting positions for each agent (float)
GOAL_POSITIONS = np.array([[90.0, 90.0], [90.0, 10.0], [10.0, 90.0]])  # Goal positions for each agent (float)
MAX_SPEED = 2.0  # Maximum speed of agents
TIME_STEP = 0.1  # Time step for simulation
GOAL_RADIUS = 1.0  # Radius to consider the goal reached
AGENT_RADIUS = 2.0  # Radius of each agent
MIN_SAFE_DISTANCE = 5.0  # Minimum safe distance between agents
OBSTACLES = [  # List of obstacles as (x, y, radius)
    (40, 40, 10),
    (60, 60, 10),
    (20, 80, 10)
]

# Check if a point is in collision with obstacles
def is_collision_free(point):
    for (ox, oy, rad) in OBSTACLES:
        if np.linalg.norm(point - np.array([ox, oy])) < rad + AGENT_RADIUS:
            return False
    return True

# Compute the preferred velocity toward the goal
def compute_preferred_velocity(position, goal):
    direction = goal - position
    distance = np.linalg.norm(direction)
    if distance > MAX_SPEED:
        direction = (direction / distance) * MAX_SPEED
    return direction

# Compute the RVO velocity for an agent
def compute_rvo_velocity(position, preferred_velocity, other_positions, other_velocities):
    rvo_velocity = preferred_velocity.copy()
    for i in range(len(other_positions)):
        relative_position = other_positions[i] - position
        relative_velocity = other_velocities[i] - preferred_velocity
        distance = np.linalg.norm(relative_position)
        if distance < MIN_SAFE_DISTANCE:
            # Adjust velocity to avoid collision
            avoidance_direction = -relative_position / distance
            rvo_velocity += avoidance_direction * MAX_SPEED
    return rvo_velocity

# Simulate RVO for all agents
def simulate_rvo(start_positions, goal_positions):
    positions = start_positions.copy()
    velocities = np.zeros_like(start_positions)
    paths = [[] for _ in range(NUM_AGENTS)]

    for _ in range(1000):  # Maximum simulation steps
        for i in range(NUM_AGENTS):
            paths[i].append(positions[i].copy())

        # Check if all agents have reached their goals
        if all(np.linalg.norm(positions[i] - goal_positions[i]) < GOAL_RADIUS for i in range(NUM_AGENTS)):
            print("All agents reached their goals!")
            break

        # Compute velocities for all agents
        new_velocities = np.zeros_like(velocities)
        for i in range(NUM_AGENTS):
            preferred_velocity = compute_preferred_velocity(positions[i], goal_positions[i])
            other_positions = np.delete(positions, i, axis=0)
            other_velocities = np.delete(velocities, i, axis=0)
            new_velocities[i] = compute_rvo_velocity(positions[i], preferred_velocity, other_positions, other_velocities)

        # Update positions and velocities
        velocities = new_velocities
        positions += velocities * TIME_STEP

        # Ensure positions are within the map and collision-free
        for i in range(NUM_AGENTS):
            positions[i] = np.clip(positions[i], [0, 0], MAP_SIZE)
            if not is_collision_free(positions[i]):
                positions[i] -= velocities[i] * TIME_STEP  # Revert to previous position

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
        path = np.array(paths[i])
        plt.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=2, label=f"Agent {i} Path")

    # Plot start and goal positions
    for i in range(NUM_AGENTS):
        plt.scatter(start_positions[i][0], start_positions[i][1], color=colors[i], marker='o', s=100, label=f"Agent {i} Start")
        plt.scatter(goal_positions[i][0], goal_positions[i][1], color=colors[i], marker='x', s=100, label=f"Agent {i} Goal")

    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])
    plt.legend()
    plt.title("RVO for 3 Agents with Inter-Agent Collision Avoidance")
    plt.show()

# Run RVO simulation and visualize
paths = simulate_rvo(START_POSITIONS, GOAL_POSITIONS)
visualize(paths, START_POSITIONS, GOAL_POSITIONS)