import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Agent:
    def __init__(self, id, start, goal, radius=0.3, max_speed=1.0):
        self.id = id
        self.position = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.radius = radius
        self.max_speed = max_speed

    def compute_preferred_velocity(self):
        """Compute the velocity towards the goal (limited by max speed)."""
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance  # Normalize
            return direction * min(distance, self.max_speed)
        return np.array([0.0, 0.0])

    def update_position(self, new_velocity, dt=0.1):
        """Update the agent's position using the new velocity."""
        self.velocity = new_velocity
        self.position += self.velocity * dt


class MovingObstacle:
    def __init__(self, start, velocity, radius=0.4):
        self.position = np.array(start, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius

    def update_position(self, dt=0.1):
        """Move the obstacle based on its velocity."""
        self.position += self.velocity * dt


def orca_velocity(agent, neighbors, moving_obstacles, time_horizon=1.0):
    """
    Compute an ORCA-based velocity that avoids collisions with other agents and moving obstacles.
    """
    preferred_velocity = agent.compute_preferred_velocity()
    constraints = []

    # Avoid other agents
    for other in neighbors:
        relative_position = other.position - agent.position
        relative_velocity = agent.velocity - other.velocity
        dist_sq = np.dot(relative_position, relative_position)
        combined_radius = agent.radius + other.radius
        combined_radius_sq = combined_radius ** 2

        if dist_sq > combined_radius_sq:
            time_to_collision = np.dot(relative_position, relative_velocity) / dist_sq
            if time_to_collision < time_horizon:
                constraint_dir = relative_position / np.sqrt(dist_sq)
                projection = np.dot(relative_velocity, constraint_dir)
                avoidance_velocity = constraint_dir * projection
                constraints.append(avoidance_velocity)

    # Avoid moving obstacles
    for obs in moving_obstacles:
        relative_position = obs.position - agent.position
        relative_velocity = agent.velocity - obs.velocity
        dist_sq = np.dot(relative_position, relative_position)
        combined_radius = agent.radius + obs.radius
        combined_radius_sq = combined_radius ** 2

        if dist_sq > combined_radius_sq:
            time_to_collision = np.dot(relative_position, relative_velocity) / dist_sq
            if time_to_collision < time_horizon:
                constraint_dir = relative_position / np.sqrt(dist_sq)
                projection = np.dot(relative_velocity, constraint_dir)
                avoidance_velocity = constraint_dir * projection
                constraints.append(avoidance_velocity)

    # Adjust velocity to satisfy constraints
    new_velocity = preferred_velocity
    for constraint in constraints:
        new_velocity += constraint * 0.5  # Adjust velocity slightly to avoid collisions

    # Limit velocity to max speed
    speed = np.linalg.norm(new_velocity)
    if speed > agent.max_speed:
        new_velocity = (new_velocity / speed) * agent.max_speed

    return new_velocity


def simulate(agents, moving_obstacles, steps=100, dt=0.1):
    """Simulates agents and moving obstacles using ORCA."""
    positions = {agent.id: [agent.position.copy()] for agent in agents}
    obstacle_positions = [obs.position.copy() for obs in moving_obstacles]

    for _ in range(steps):
        for agent in agents:
            neighbors = [a for a in agents if a.id != agent.id]
            new_velocity = orca_velocity(agent, neighbors, moving_obstacles)
            agent.update_position(new_velocity, dt)
            positions[agent.id].append(agent.position.copy())

        for obs in moving_obstacles:
            obs.update_position(dt)
            obstacle_positions.append(obs.position.copy())

        if all(np.linalg.norm(agent.position - agent.goal) < 0.1 for agent in agents):
            break

    return positions, obstacle_positions


def animate_simulation(agents, positions, moving_obstacles, obstacle_positions):
    """Animate the agent and obstacle movement over time."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Agent ORCA with Moving Obstacles")
    ax.grid()

    # Draw agent goals
    for agent in agents:
        ax.scatter(*agent.goal, marker='x', color='red', s=100, label=f"Goal {agent.id}")

    # Agent markers
    agent_patches = [plt.Circle(agent.position, agent.radius, color=np.random.rand(3, )) for agent in agents]
    for patch in agent_patches:
        ax.add_patch(patch)

    # Moving obstacles
    obstacle_patches = [plt.Circle(obs.position, obs.radius, color='black') for obs in moving_obstacles]
    for patch in obstacle_patches:
        ax.add_patch(patch)

    def update(frame):
        for i, agent in enumerate(agents):
            pos = positions[agent.id][frame]
            agent_patches[i].center = pos

        for j, obs in enumerate(moving_obstacles):
            pos = obstacle_positions[frame + j * len(positions[agents[0].id]) // len(moving_obstacles)]
            obstacle_patches[j].center = pos

        return agent_patches + obstacle_patches

    ani = animation.FuncAnimation(fig, update, frames=len(next(iter(positions.values()))), interval=100, blit=False)
    plt.show()


# Example usage
agents = [
    Agent(0, [0, 0], [5, 5]),
    Agent(1, [5, 0], [0, 5]),
    Agent(2, [0, 5], [5, 0])
]

# Define moving obstacles
moving_obstacles = [
    MovingObstacle([2.5, 2.5], [0.02, -0.02]),
    MovingObstacle([1.5, 3.5], [-0.02, 0.02])
]

positions, obstacle_positions = simulate(agents, moving_obstacles, steps=200)
animate_simulation(agents, positions, moving_obstacles, obstacle_positions)
