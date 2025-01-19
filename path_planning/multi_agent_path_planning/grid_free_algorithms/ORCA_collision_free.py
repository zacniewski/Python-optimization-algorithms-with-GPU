import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Agent:
    def __init__(self, id, start, goal, radius=0.3, max_speed=1.0, goal_tolerance=0.1):
        self.id = id
        self.position = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.radius = radius
        self.max_speed = max_speed
        self.goal_tolerance = goal_tolerance

    def compute_preferred_velocity(self):
        """Compute the velocity towards the goal (limited by max speed)."""
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance < self.goal_tolerance:
            return np.array([0.0, 0.0])  # Stop moving
        direction = direction / distance  # Normalize
        return direction * min(distance, self.max_speed)

    def update_position(self, new_velocity, dt=0.1):
        """Update the agent's position using the new velocity."""
        self.velocity = new_velocity
        self.position += self.velocity * dt


def orca_velocity(agent, neighbors, time_horizon=1.0):
    """
    Compute an ORCA-based velocity that strictly avoids collisions.
    """
    preferred_velocity = agent.compute_preferred_velocity()
    if np.linalg.norm(preferred_velocity) == 0:
        return np.array([0.0, 0.0])  # Stop if at goal

    constraints = []

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

    # Enforce ORCA constraints using velocity projection
    new_velocity = preferred_velocity
    for constraint in constraints:
        if np.linalg.norm(constraint) > 1e-6:  # ✅ Fix: Prevent division by zero
            projection = np.dot(new_velocity, constraint) / np.dot(constraint, constraint)
            if projection > 0:
                new_velocity -= constraint * projection

    # Ensure velocity does not exceed max speed
    speed = np.linalg.norm(new_velocity)
    if speed > agent.max_speed:
        new_velocity = (new_velocity / speed) * agent.max_speed

    return new_velocity


def simulate(agents, steps=100, dt=0.1):
    """Simulates agents moving towards their goals using ORCA constraints."""
    positions = {agent.id: [agent.position.copy()] for agent in agents}

    for _ in range(steps):
        for agent in agents:
            if np.linalg.norm(agent.position - agent.goal) < agent.goal_tolerance:
                continue  # Stop updating if at goal

            neighbors = [a for a in agents if a.id != agent.id]
            new_velocity = orca_velocity(agent, neighbors)
            agent.update_position(new_velocity, dt)
            positions[agent.id].append(agent.position.copy())

        if all(np.linalg.norm(agent.position - agent.goal) < agent.goal_tolerance for agent in agents):
            break  # Stop simulation if all agents reach their goals

    return positions


def animate_simulation(agents, positions):
    """Animate the agent movement over time with path visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Collision-Free Multi-Agent ORCA with Paths")
    ax.grid()

    # Draw agent goals
    for agent in agents:
        ax.scatter(*agent.goal, marker='x', color='red', s=100, label=f"Goal {agent.id}")

    # Agent markers
    agent_patches = [plt.Circle(agent.position, agent.radius, color=np.random.rand(3, )) for agent in agents]
    for patch in agent_patches:
        ax.add_patch(patch)

    # Paths (lines for each agent)
    paths = [ax.plot([], [], linestyle='--', color=patch.get_facecolor())[0] for patch in agent_patches]

    def update(frame):
        for i, agent in enumerate(agents):
            pos = positions[agent.id][frame]
            agent_patches[i].center = pos

            # Update path
            path_data = np.array(positions[agent.id][:frame + 1])
            paths[i].set_data(path_data[:, 0], path_data[:, 1])

        return agent_patches + paths

    ani = animation.FuncAnimation(fig, update, frames=len(next(iter(positions.values()))), interval=100, blit=False)
    plt.show()


# Example usage
agents = [
    Agent(0, [0, 0], [5, 5]),
    Agent(1, [5, 0], [0, 5]),
    Agent(2, [0, 5], [5, 0])
]

positions = simulate(agents, steps=600)
animate_simulation(agents, positions)
