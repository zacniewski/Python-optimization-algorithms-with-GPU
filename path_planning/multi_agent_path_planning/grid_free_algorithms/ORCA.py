import numpy as np
import matplotlib.pyplot as plt

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

def orca_velocity(agent, neighbors, time_horizon=1.0):
    """
    Compute an ORCA-based velocity that avoids collisions with neighbors.
    :param agent: The current agent.
    :param neighbors: List of other agents.
    :param time_horizon: Lookahead time to avoid collisions.
    :return: New velocity for the agent.
    """
    preferred_velocity = agent.compute_preferred_velocity()
    constraints = []

    for other in neighbors:
        relative_position = other.position - agent.position
        relative_velocity = agent.velocity - other.velocity
        dist_sq = np.dot(relative_position, relative_position)
        combined_radius = agent.radius + other.radius
        combined_radius_sq = combined_radius ** 2

        if dist_sq > combined_radius_sq:
            # Compute velocity obstacle
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

def simulate(agents, steps=100, dt=0.1):
    """Simulates agents moving towards their goals using ORCA."""
    positions = {agent.id: [agent.position.copy()] for agent in agents}

    for _ in range(steps):
        for agent in agents:
            neighbors = [a for a in agents if a.id != agent.id]
            new_velocity = orca_velocity(agent, neighbors)
            agent.update_position(new_velocity, dt)
            positions[agent.id].append(agent.position.copy())

        # Stop if all agents reach their goals
        if all(np.linalg.norm(agent.position - agent.goal) < 0.1 for agent in agents):
            break

    return positions

def plot_paths(agents, positions):
    """Plot the movement paths of all agents."""
    plt.figure(figsize=(6,6))
    for agent in agents:
        path = np.array(positions[agent.id])
        plt.plot(path[:,0], path[:,1], label=f"Agent {agent.id}")
        plt.scatter(agent.position[0], agent.position[1], marker='o', label=f"Goal {agent.id}")
        plt.scatter(agent.goal[0], agent.goal[1], marker='x', color='red')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Multi-Agent Path Planning with ORCA")
    plt.grid()
    plt.show()

# Example usage
agents = [
    Agent(0, [0, 0], [5, 5]),
    Agent(1, [5, 0], [0, 5]),
    Agent(2, [0, 5], [5, 0])
]

positions = simulate(agents, steps=100)
plot_paths(agents, positions)
