import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

### --- RRT* PATH PLANNING --- ###
class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.cost = 0

def get_nearest_node(tree, point):
    return min(tree, key=lambda node: np.linalg.norm(node.position - point))

def steer(from_node, to_point, step_size=0.5):
    direction = to_point - from_node.position
    distance = np.linalg.norm(direction)
    if distance > step_size:
        direction = (direction / distance) * step_size
    return Node(from_node.position + direction, from_node)

def get_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Reverse the path

def rrt_star(start, goal, x_bounds, y_bounds, max_iter=500, step_size=0.5):
    tree = [Node(start)]
    for _ in range(max_iter):
        random_point = np.array([random.uniform(*x_bounds), random.uniform(*y_bounds)])
        nearest_node = get_nearest_node(tree, random_point)
        new_node = steer(nearest_node, random_point, step_size)

        if not np.any(np.isnan(new_node.position)):  # Ensure valid point
            new_node.cost = nearest_node.cost + np.linalg.norm(new_node.position - nearest_node.position)
            tree.append(new_node)

            if np.linalg.norm(new_node.position - goal) < step_size:
                return get_path(new_node)
    return None  # Path not found

### --- ORCA COLLISION AVOIDANCE --- ###
class Agent:
    def __init__(self, id, start, goal, radius=0.3, max_speed=1.0, goal_tolerance=0.1):
        self.id = id
        self.position = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.radius = radius
        self.max_speed = max_speed
        self.goal_tolerance = goal_tolerance
        self.path = []

    def compute_preferred_velocity(self):
        """Follow RRT* path smoothly."""
        if not self.path:
            return np.array([0.0, 0.0])  # Stop if no path

        next_waypoint = self.path[0]
        direction = next_waypoint - self.position
        distance = np.linalg.norm(direction)

        if distance < self.goal_tolerance:
            self.path.pop(0)  # Move to next waypoint
            if not self.path:
                return np.array([0.0, 0.0])  # Stop at goal

        if distance > 1e-6:  # ✅ Fix: Prevent division by zero
            direction = direction / distance  # Normalize
        else:
            return np.array([0.0, 0.0])  # Stop moving

        return direction * min(distance, self.max_speed)

    def update_position(self, new_velocity, dt=0.1):
        """Update agent's position using new velocity."""
        self.velocity = new_velocity
        self.position += self.velocity * dt

def orca_velocity(agent, neighbors, time_horizon=1.0):
    """Compute ORCA-based velocity for collision-free movement."""
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

    new_velocity = preferred_velocity
    for constraint in constraints:
        if np.linalg.norm(constraint) > 1e-6:  # Prevent division by zero
            projection = np.dot(new_velocity, constraint) / np.dot(constraint, constraint)
            if projection > 0:
                new_velocity -= constraint * projection

    speed = np.linalg.norm(new_velocity)
    if speed > agent.max_speed:
        new_velocity = (new_velocity / speed) * agent.max_speed

    return new_velocity

def simulate(agents, steps=100, dt=0.1):
    """Simulates agents moving along RRT* paths with ORCA avoidance."""
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

### --- ANIMATION --- ###
def animate_simulation(agents, positions):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Agent Pathfinding: ORCA + RRT*")
    ax.grid()

    # Draw agent goals
    for agent in agents:
        ax.scatter(*agent.goal, marker='x', color='red', s=100, label=f"Goal {agent.id}")

    agent_patches = [plt.Circle(agent.position, agent.radius, color=np.random.rand(3,)) for agent in agents]
    for patch in agent_patches:
        ax.add_patch(patch)

    paths = [ax.plot([], [], linestyle='--', color=patch.get_facecolor())[0] for patch in agent_patches]

    def update(frame):
        for i, agent in enumerate(agents):
            pos = positions[agent.id][frame]
            agent_patches[i].center = pos
            path_data = np.array(positions[agent.id][:frame+1])
            paths[i].set_data(path_data[:, 0], path_data[:, 1])
        return agent_patches + paths

    ani = animation.FuncAnimation(fig, update, frames=len(next(iter(positions.values()))), interval=100, blit=False)
    plt.show()

# --- RUN THE HYBRID ALGORITHM --- #
x_bounds = (0, 5)
y_bounds = (0, 5)

agents = [
    Agent(0, [0, 0], [4, 4]),
    Agent(1, [4, 0], [0, 4]),
    Agent(2, [0, 4], [4, 0])
]

# Compute paths using RRT*
for agent in agents:
    agent.path = rrt_star(agent.position, agent.goal, x_bounds, y_bounds)

positions = simulate(agents, steps=200)
animate_simulation(agents, positions)
