import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from scipy.spatial.distance import euclidean

class Agent:
    def __init__(self, id, start_pos, goal_pos, color, size=0.3):
        self.id = id
        self.position = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.color = color
        self.size = size
        self.path = []
        self.max_speed = 2.0
        self.max_force = 0.5
        self.arrival_threshold = 0.5
        
    def update(self, agents, obstacles, restricted_areas, weather, dt):
        # Calculate steering forces
        goal_force = self.seek(self.goal)
        avoid_agent_force = self.avoid_agents(agents)
        avoid_obstacle_force = self.avoid_obstacles(obstacles)
        avoid_restricted_force = self.avoid_restricted_areas(restricted_areas)
        
        # Apply weather effect
        weather_effect = np.array([weather['wind_x'], weather['wind_y']]) * 0.3
        
        # Combine all forces
        total_force = (goal_force * 1.5 + avoid_agent_force * 1.2 + 
                       avoid_obstacle_force * 1.5 + avoid_restricted_force * 1.5 + 
                       weather_effect)
        
        # Limit force
        total_force = self.limit_force(total_force)
        
        # Update velocity and position
        self.velocity += total_force * dt
        self.velocity = self.limit_velocity(self.velocity)
        self.position += self.velocity * dt
        
        # Record path (for visualization)
        self.path.append(self.position.copy())
        if len(self.path) > 100:  # Limit path length
            self.path.pop(0)
    
    def seek(self, target):
        desired_velocity = (target - self.position)
        distance = np.linalg.norm(desired_velocity)
        
        if distance < self.arrival_threshold:
            return np.zeros(2)
        
        desired_velocity = desired_velocity / distance * self.max_speed
        
        # Slow down when approaching the goal
        if distance < 5.0:
            desired_velocity = desired_velocity * (distance / 5.0)
        
        steer = desired_velocity - self.velocity
        return self.limit_force(steer)
    
    def avoid_agents(self, agents):
        avoidance_force = np.zeros(2)
        avoidance_distance = self.size * 3
        
        for agent in agents:
            if agent.id == self.id:
                continue
                
            to_other = self.position - agent.position
            distance = np.linalg.norm(to_other)
            
            if distance < avoidance_distance and distance > 0:
                strength = min(1.0, (avoidance_distance - distance) / avoidance_distance)
                avoidance_force += (to_other / distance) * strength * self.max_force * 2
        
        return avoidance_force
    
    def avoid_obstacles(self, obstacles):
        avoidance_force = np.zeros(2)
        avoidance_distance = self.size * 4
        
        for obstacle in obstacles:
            to_obstacle = obstacle['position'] - self.position
            distance = np.linalg.norm(to_obstacle)
            
            if distance < (obstacle['radius'] + avoidance_distance):
                if distance <= obstacle['radius']:
                    # Emergency avoidance if inside obstacle
                    if distance == 0:
                        # Random direction if exactly at center (unlikely)
                        escape_dir = np.random.uniform(-1, 1, 2)
                        escape_dir = escape_dir / np.linalg.norm(escape_dir)
                    else:
                        escape_dir = -to_obstacle / distance
                    avoidance_force += escape_dir * self.max_force * 3
                else:
                    # Normal avoidance
                    strength = 1.0 - (distance - obstacle['radius']) / avoidance_distance
                    avoidance_force += (-to_obstacle / distance) * strength * self.max_force * 2
        
        return avoidance_force
    
    def avoid_restricted_areas(self, restricted_areas):
        avoidance_force = np.zeros(2)
        avoidance_distance = self.size * 5
        
        for area in restricted_areas:
            # Check if agent is inside the restricted area
            if (area['x'] - area['width']/2 <= self.position[0] <= area['x'] + area['width']/2 and
                area['y'] - area['height']/2 <= self.position[1] <= area['y'] + area['height']/2):
                
                # Find closest edge
                left_dist = self.position[0] - (area['x'] - area['width']/2)
                right_dist = (area['x'] + area['width']/2) - self.position[0]
                bottom_dist = self.position[1] - (area['y'] - area['height']/2)
                top_dist = (area['y'] + area['height']/2) - self.position[1]
                
                min_dist = min(left_dist, right_dist, bottom_dist, top_dist)
                
                if min_dist == left_dist:
                    escape_dir = np.array([-1, 0])
                elif min_dist == right_dist:
                    escape_dir = np.array([1, 0])
                elif min_dist == bottom_dist:
                    escape_dir = np.array([0, -1])
                else:
                    escape_dir = np.array([0, 1])
                
                avoidance_force += escape_dir * self.max_force * 3
            
            # Also avoid getting too close to restricted areas
            else:
                # Calculate distance to area boundaries
                closest_x = max(area['x'] - area['width']/2, min(self.position[0], area['x'] + area['width']/2))
                closest_y = max(area['y'] - area['height']/2, min(self.position[1], area['y'] + area['height']/2))
                
                to_area = np.array([closest_x, closest_y]) - self.position
                distance = np.linalg.norm(to_area)
                
                if distance < avoidance_distance:
                    strength = 1.0 - distance / avoidance_distance
                    avoidance_force += (-to_area / distance) * strength * self.max_force
        
        return avoidance_force
    
    def limit_velocity(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity / speed * self.max_speed
        return velocity
    
    def limit_force(self, force):
        force_norm = np.linalg.norm(force)
        if force_norm > self.max_force:
            return force / force_norm * self.max_force
        return force
    
    def reached_goal(self):
        return np.linalg.norm(self.position - self.goal) < self.arrival_threshold

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.obstacles = []
        self.restricted_areas = []
        self.required_areas = []
        self.weather = {'wind_x': 0, 'wind_y': 0}
        self.time = 0
        
    def add_agent(self, agent):
        self.agents.append(agent)
        
    def add_obstacle(self, position, radius):
        self.obstacles.append({'position': np.array(position), 'radius': radius})
        
    def add_restricted_area(self, x, y, width, height):
        self.restricted_areas.append({'x': x, 'y': y, 'width': width, 'height': height})
        
    def add_required_area(self, x, y, width, height):
        self.required_areas.append({'x': x, 'y': y, 'width': width, 'height': height})
        
    def update_weather(self):
        # Change weather gradually and randomly
        self.time += 1
        if self.time % 20 == 0:  # Change weather every 20 updates
            self.weather['wind_x'] = np.random.uniform(-1, 1)
            self.weather['wind_y'] = np.random.uniform(-1, 1)
        
    def update(self, dt):
        self.update_weather()
        
        for agent in self.agents:
            agent.update(self.agents, self.obstacles, self.restricted_areas, self.weather, dt)
            
            # Check if agent needs to visit required areas
            for area in self.required_areas:
                if (area['x'] - area['width']/2 <= agent.position[0] <= area['x'] + area['width']/2 and
                    area['y'] - area['height']/2 <= agent.position[1] <= area['y'] + area['height']/2):
                    # This agent has visited this area
                    pass

def visualize(env, frames=500):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Draw obstacles
    for obstacle in env.obstacles:
        circle = plt.Circle(obstacle['position'], obstacle['radius'], color='gray', alpha=0.7)
        ax.add_patch(circle)
    
    # Draw restricted areas
    for area in env.restricted_areas:
        rect = patches.Rectangle(
            (area['x'] - area['width']/2, area['y'] - area['height']/2),
            area['width'], area['height'],
            linewidth=1, edgecolor='r', facecolor='red', alpha=0.2
        )
        ax.add_patch(rect)
    
    # Draw required areas
    for area in env.required_areas:
        rect = patches.Rectangle(
            (area['x'] - area['width']/2, area['y'] - area['height']/2),
            area['width'], area['height'],
            linewidth=1, edgecolor='g', facecolor='green', alpha=0.2
        )
        ax.add_patch(rect)
    
    # Initialize agent artists
    agent_artists = []
    path_artists = []
    for agent in env.agents:
        agent_circle = plt.Circle(agent.position, agent.size, color=agent.color, alpha=0.8)
        agent_artists.append(ax.add_patch(agent_circle))
        
        # Goal marker
        goal_marker = plt.Circle(agent.goal, 0.2, color=agent.color, alpha=0.3)
        ax.add_patch(goal_marker)
        
        # Path line
        path_line, = ax.plot([], [], color=agent.color, alpha=0.5, linewidth=1)
        path_artists.append(path_line)
    
    # Weather text
    weather_text = ax.text(env.width * 0.02, env.height * 0.95, '', fontsize=10)
    
    def init():
        return agent_artists + path_artists + [weather_text]
    
    def animate(i):
        env.update(0.1)  # Update with small time step
        
        for j, agent in enumerate(env.agents):
            agent_artists[j].center = agent.position
            if len(agent.path) > 1:
                path_artists[j].set_data([p[0] for p in agent.path], [p[1] for p in agent.path])
        
        # Update weather display
        wind_speed = np.linalg.norm([env.weather['wind_x'], env.weather['wind_y']])
        wind_dir = np.arctan2(env.weather['wind_y'], env.weather['wind_x']) * 180 / np.pi
        weather_text.set_text(f'Wind: {wind_speed:.1f} m/s, Direction: {wind_dir:.0f}°')
        
        return agent_artists + path_artists + [weather_text]
    
    anim = FuncAnimation(fig, animate, frames=frames, init_func=init,
                         blit=True, interval=50)
    plt.title('Multi-Agent Path Planning with Continuous Space')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()
    return anim

# Create environment
env = Environment(width=20, height=20)

# Add obstacles
env.add_obstacle(position=(5, 5), radius=1.5)
env.add_obstacle(position=(15, 5), radius=1.0)
env.add_obstacle(position=(5, 15), radius=1.0)
env.add_obstacle(position=(15, 15), radius=1.5)

# Add restricted areas (no-go zones)
env.add_restricted_area(x=10, y=10, width=3, height=3)
env.add_restricted_area(x=3, y=17, width=2, height=2)
env.add_restricted_area(x=17, y=3, width=2, height=2)

# Add required areas (must visit zones)
env.add_required_area(x=3, y=3, width=2, height=2)
env.add_required_area(x=17, y=17, width=2, height=2)

# Add agents with different start and goal positions
env.add_agent(Agent(0, start_pos=(2, 2), goal_pos=(18, 18), color='blue'))
env.add_agent(Agent(1, start_pos=(18, 2), goal_pos=(2, 18), color='red'))
env.add_agent(Agent(2, start_pos=(2, 18), goal_pos=(18, 2), color='green'))
env.add_agent(Agent(3, start_pos=(18, 18), goal_pos=(2, 2), color='purple'))

# Visualize the simulation
anim = visualize(env, frames=500)