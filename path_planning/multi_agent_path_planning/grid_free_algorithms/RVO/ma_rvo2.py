import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Parameters
num_agents = 4
radius = 0.2  # Smaller agents
max_speed = 1.0  # Increased max speed for better navigation
time_horizon = 5.0
time_step = 0.1
goal_tolerance = 0.2
repulsion_strength = 2.0  # Strength of repulsion from obstacles

# Agent and obstacle definitions
agents = [
    {'position': np.array([0.0, 0.0]), 'velocity': np.array([0.0, 0.0]), 'goal': np.array([5.0, 5.0])},
    {'position': np.array([5.0, 0.0]), 'velocity': np.array([0.0, 0.0]), 'goal': np.array([0.0, 5.0])},
    {'position': np.array([0.0, 5.0]), 'velocity': np.array([0.0, 0.0]), 'goal': np.array([5.0, 0.0])},
    {'position': np.array([5.0, 5.0]), 'velocity': np.array([0.0, 0.0]), 'goal': np.array([0.0, 0.0])}
]

# Adjusted obstacle positions and sizes
obstacles = [
    {'position': np.array([2.0, 2.0]), 'radius': 0.5},  # Smaller obstacle
    {'position': np.array([1.0, 3.5]), 'radius': 0.4},  # Adjusted position
    {'position': np.array([4.0, 1.5]), 'radius': 0.4}   # Adjusted position
]

# Helper functions
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-5:
        return np.zeros_like(v)
    return v / norm

def compute_rvo_velocity(agent, other_agents, obstacles):
    v_pref = normalize(agent['goal'] - agent['position']) * max_speed
    vo_velocities = []

    # Avoid other agents
    for other in other_agents:
        if np.array_equal(other['position'], agent['position']):
            continue
        relative_position = other['position'] - agent['position']
        relative_velocity = agent['velocity'] - other['velocity']
        dist_sq = np.sum(relative_position**2)
        combined_radius = radius * 2
        if dist_sq > combined_radius**2:
            continue
        # Stronger repulsion from other agents
        repulsion_force = -normalize(relative_position) * repulsion_strength
        vo_velocities.append(repulsion_force)

    # Avoid obstacles with stronger repulsion
    for obstacle in obstacles:
        relative_position = obstacle['position'] - agent['position']
        dist = np.linalg.norm(relative_position)
        if dist > radius + obstacle['radius']:
            continue
        # Stronger repulsion force
        repulsion_force = -normalize(relative_position) * repulsion_strength
        vo_velocities.append(repulsion_force)

    # Compute RVO velocity
    if len(vo_velocities) == 0:
        return v_pref
    vo_velocity = np.mean(vo_velocities, axis=0)
    return v_pref + vo_velocity

# Simulation loop
fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_aspect('equal')

for step in range(200):  # Increased simulation steps
    plt.cla()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')

    # Update agent positions and velocities
    for i, agent in enumerate(agents):
        # Stop if the agent is close to its goal
        if np.linalg.norm(agent['position'] - agent['goal']) < goal_tolerance:
            agent['velocity'] = np.array([0.0, 0.0])
            continue

        other_agents = [a for j, a in enumerate(agents) if j != i]
        new_velocity = compute_rvo_velocity(agent, other_agents, obstacles)
        agent['velocity'] = new_velocity
        agent['position'] += agent['velocity'] * time_step

        # Draw agents
        circle = Circle(agent['position'], radius, color='blue', alpha=0.5)
        ax.add_patch(circle)
        ax.plot([agent['position'][0], agent['goal'][0]], [agent['position'][1], agent['goal'][1]], 'k--')

    # Draw obstacles
    for obstacle in obstacles:
        circle = Circle(obstacle['position'], obstacle['radius'], color='red', alpha=0.5)
        ax.add_patch(circle)

    plt.pause(0.1)

plt.show()
