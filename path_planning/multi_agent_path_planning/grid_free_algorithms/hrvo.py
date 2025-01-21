import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # For RRT*
import random


class RRTStar:
    def __init__(self, start, goal, obstacles, x_bounds, y_bounds, step_size=5, max_iter=1000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.graph = nx.Graph()
        self.graph.add_node(start)

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_random_point(self):
        return (random.uniform(*self.x_bounds), random.uniform(*self.y_bounds))

    def get_nearest_node(self, point):
        return min(self.graph.nodes, key=lambda n: self.distance(n, point))

    def steer(self, from_node, to_point):
        vec = np.array(to_point) - np.array(from_node)
        dist = np.linalg.norm(vec)
        if dist < self.step_size:
            return to_point
        return tuple(np.array(from_node) + (vec / dist) * self.step_size)

    def is_collision_free(self, p1, p2):
        for obs in self.obstacles:
            if self.distance(obs, p1) < 5 or self.distance(obs, p2) < 5:
                return False
        return True

    def find_path(self):
        for _ in range(self.max_iter):
            rand_point = self.get_random_point()
            nearest_node = self.get_nearest_node(rand_point)
            new_node = self.steer(nearest_node, rand_point)

            if self.is_collision_free(nearest_node, new_node):
                self.graph.add_node(new_node)
                self.graph.add_edge(nearest_node, new_node, weight=self.distance(nearest_node, new_node))
                if self.distance(new_node, self.goal) < self.step_size:
                    self.graph.add_node(self.goal)
                    self.graph.add_edge(new_node, self.goal, weight=self.distance(new_node, self.goal))
                    path = nx.shortest_path(self.graph, self.start, self.goal, weight='weight')
                    print(f"Path for agent from {self.start} to {self.goal}: {path}")
                    return path
        return None


class HRVO_Agent:
    def __init__(self, position, goal):
        self.position = np.array(position, dtype=np.float64)
        self.goal = np.array(goal)
        self.velocity = np.array([0.0, 0.0])
        self.traveled_path = [tuple(self.position)]  # Track movement history

    def update_velocity(self, neighbors, max_speed=1.0):
        desired_velocity = self.goal - self.position
        desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * max_speed

        avoidance_velocity = desired_velocity.copy()
        for neighbor in neighbors:
            relative_position = neighbor.position - self.position
            relative_velocity = neighbor.velocity - self.velocity

            if np.linalg.norm(relative_position) < 10.0:  # Adjusted HRVO avoidance threshold
                avoidance_direction = -relative_position / np.linalg.norm(relative_position)
                avoidance_velocity += avoidance_direction * max_speed * 0.5

        self.velocity = avoidance_velocity / np.linalg.norm(avoidance_velocity) * max_speed if np.linalg.norm(
            avoidance_velocity) > 0 else self.velocity
        desired_velocity = self.goal - self.position
        desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * max_speed
        for neighbor in neighbors:
            relative_position = neighbor.position - self.position
            relative_velocity = neighbor.velocity - self.velocity
            if np.linalg.norm(relative_position) < 5.0:
                avoidance_force = -relative_position / np.linalg.norm(relative_position) * max_speed
                desired_velocity += avoidance_force
        self.velocity = desired_velocity / np.linalg.norm(desired_velocity) * max_speed if np.linalg.norm(
            desired_velocity) > 0 else self.velocity

    def move(self, obstacles):
        new_position = self.position + self.velocity

        if all(np.linalg.norm(new_position - np.array(obs)) >= 5 for obs in obstacles):
            self.position = new_position
        self.traveled_path.append(tuple(self.position))  # Store movement history


def simulate_agents(agents, paths, obstacles):
    for step in range(500):

        for obs in obstacles:
            plt.plot(obs[0], obs[1], 'ko', markersize=10)  # Plot obstacles

        for i, agent in enumerate(agents):
            agent.update_velocity([a for j, a in enumerate(agents) if j != i])
            agent.move(obstacles)
            plt.plot(agent.position[0], agent.position[1], 'bo', markersize=8)
            plt.plot(agent.goal[0], agent.goal[1], 'ro', markersize=8)
            if agent.traveled_path:
                path_x, path_y = zip(*agent.traveled_path)
                plt.plot(path_x, path_y, 'g-', linewidth=1)  # Show real-time movement
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.pause(0.01)
    plt.show(block=True)


if __name__ == "__main__":
    obstacles = [(30, 30), (70, 70), (50, 50), (40, 60), (60, 40), (20, 50)]
    agents_positions = [(10, 10), (20, 90), (80, 20)]
    goals = [(90, 90), (10, 10), (50, 80)]

    paths = []
    agents = []
    for agent_pos, goal in zip(agents_positions, goals):
        rrt_star = RRTStar(agent_pos, goal, obstacles, (0, 100), (0, 100))
        path = rrt_star.find_path()
        paths.append(path if path else [agent_pos])
        agents.append(HRVO_Agent(agent_pos, goal))

        simulate_agents(agents, paths, obstacles)
