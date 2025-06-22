import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from scipy.spatial.distance import euclidean
import matplotlib.animation as animation

"""
Multi-Agent Path Planning with Environmental Factors

This script implements a multi-agent path planning system in continuous space with environmental
factors like wind and turbulence. The simulation includes obstacles, restricted areas that agents
should avoid, and required areas that agents must visit.

Algorithms used:
1. Steering Behaviors: The agents use Craig Reynolds' steering behaviors (seek, avoid) to navigate
   through the environment. This is a reactive approach where agents calculate forces based on their
   surroundings and update their velocities accordingly.
2. Potential Field Method: The avoidance behaviors implement a form of artificial potential fields
   where obstacles and restricted areas generate repulsive forces, while goals and required areas
   generate attractive forces.
3. Force Accumulation: Multiple steering forces are calculated separately and then combined with
   different weights to determine the final movement direction.

Color coding in visualization:
- Blue agent: Agent 0 (starts at bottom-left)
- Red agent: Agent 1 (starts at bottom-right)
- Green agent: Agent 2 (starts at top-left)
- Purple agent: Agent 3 (starts at top-right)
- Gray circles: Obstacles that agents must avoid
- Red rectangles: Restricted areas (no-go zones)
- Green rectangles: Required areas (must-visit zones)
  - Light green: Required areas that have been visited
  - Dark green: Required areas that have not been visited yet
- Blue arrow: Current wind direction and strength
"""

class Agent:
    """
    Agent class for path planning in continuous space.

    Each agent has a start position, goal position, and navigates through the environment
    while avoiding obstacles, other agents, and restricted areas. Agents are also affected
    by environmental factors like wind.
    """
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
        self.visited_required_areas = set()  # Track visited required areas

    def update(self, agents, obstacles, restricted_areas, required_areas, weather, dt):
        """Update agent position based on various forces and environmental factors."""
        # Calculate steering forces
        goal_force = self.seek(self.goal)
        avoid_agent_force = self.avoid_agents(agents)
        avoid_obstacle_force = self.avoid_obstacles(obstacles)
        avoid_restricted_force = self.avoid_restricted_areas(restricted_areas)
        seek_required_force = self.seek_required_areas(required_areas)

        # Apply weather effect
        weather_effect = np.array([weather['wind_x'], weather['wind_y']]) * 0.3

        # Combine all forces
        total_force = (goal_force * 1.5 + 
                       avoid_agent_force * 1.2 + 
                       avoid_obstacle_force * 1.5 + 
                       avoid_restricted_force * 1.5 + 
                       seek_required_force * 1.0 + 
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

        # Check if agent is in any required area and mark as visited
        for i, area in enumerate(required_areas):
            if (area['x'] - area['width']/2 <= self.position[0] <= area['x'] + area['width']/2 and
                area['y'] - area['height']/2 <= self.position[1] <= area['y'] + area['height']/2):
                self.visited_required_areas.add(i)

    def seek(self, target):
        """Generate force to move toward target."""
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
        """Generate force to avoid other agents."""
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
        """Generate force to avoid obstacles."""
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
        """Generate force to avoid restricted areas."""
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

    def seek_required_areas(self, required_areas):
        """Generate force to seek required areas that haven't been visited yet."""
        seek_force = np.zeros(2)

        # Only seek required areas that haven't been visited yet
        for i, area in enumerate(required_areas):
            if i in self.visited_required_areas:
                continue

            # Calculate center of the area
            center = np.array([area['x'], area['y']])
            to_area = center - self.position
            distance = np.linalg.norm(to_area)

            # Only apply seeking force if we're not too close to the goal
            if distance > 0 and np.linalg.norm(self.goal - self.position) > 5.0:
                # Strength decreases as we get closer to the goal
                goal_distance = np.linalg.norm(self.goal - self.position)
                goal_factor = min(1.0, goal_distance / 10.0)

                # Normalize and scale
                seek_force += (to_area / distance) * self.max_force * 0.8 * goal_factor

        return seek_force

    def limit_velocity(self, velocity):
        """Limit velocity to maximum speed."""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity / speed * self.max_speed
        return velocity

    def limit_force(self, force):
        """Limit force to maximum force."""
        force_norm = np.linalg.norm(force)
        if force_norm > self.max_force:
            return force / force_norm * self.max_force
        return force

    def reached_goal(self):
        """Check if agent has reached its goal."""
        return np.linalg.norm(self.position - self.goal) < self.arrival_threshold

    def all_required_areas_visited(self, required_areas):
        """Check if agent has visited all required areas."""
        return len(self.visited_required_areas) == len(required_areas)

class Environment:
    """
    Environment class for multi-agent path planning.

    Contains agents, obstacles, restricted areas, required areas, and weather conditions.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.obstacles = []
        self.restricted_areas = []
        self.required_areas = []
        self.weather = {'wind_x': 0, 'wind_y': 0, 'turbulence': 0}
        self.time = 0

    def add_agent(self, agent):
        """Add an agent to the environment."""
        self.agents.append(agent)

    def add_obstacle(self, position, radius):
        """Add a circular obstacle to the environment."""
        self.obstacles.append({'position': np.array(position), 'radius': radius})

    def add_restricted_area(self, x, y, width, height):
        """Add a rectangular restricted area (no-go zone) to the environment."""
        self.restricted_areas.append({'x': x, 'y': y, 'width': width, 'height': height})

    def add_required_area(self, x, y, width, height):
        """Add a rectangular required area (must-visit zone) to the environment."""
        self.required_areas.append({'x': x, 'y': y, 'width': width, 'height': height})

    def update_weather(self):
        """Update weather conditions in the environment."""
        # Change weather gradually and randomly
        self.time += 1

        # Change wind every 20 updates
        if self.time % 20 == 0:
            # Wind changes gradually
            self.weather['wind_x'] = 0.7 * self.weather['wind_x'] + 0.3 * np.random.uniform(-1.5, 1.5)
            self.weather['wind_y'] = 0.7 * self.weather['wind_y'] + 0.3 * np.random.uniform(-1.5, 1.5)

            # Add some turbulence (random fluctuations)
            self.weather['turbulence'] = np.random.uniform(0, 0.5)

        # Add random turbulence to wind
        if np.random.random() < 0.1:  # 10% chance of turbulence
            turbulence_x = np.random.normal(0, self.weather['turbulence'])
            turbulence_y = np.random.normal(0, self.weather['turbulence'])
            self.weather['wind_x'] += turbulence_x
            self.weather['wind_y'] += turbulence_y

            # Limit wind strength
            wind_strength = np.sqrt(self.weather['wind_x']**2 + self.weather['wind_y']**2)
            if wind_strength > 2.0:
                self.weather['wind_x'] = self.weather['wind_x'] / wind_strength * 2.0
                self.weather['wind_y'] = self.weather['wind_y'] / wind_strength * 2.0

    def update(self, dt):
        """Update the environment and all agents."""
        self.update_weather()

        for agent in self.agents:
            agent.update(self.agents, self.obstacles, self.restricted_areas, 
                         self.required_areas, self.weather, dt)

    def all_agents_reached_goals(self):
        """Check if all agents have reached their goals."""
        return all(agent.reached_goal() for agent in self.agents)

    def all_required_areas_visited(self):
        """Check if all required areas have been visited by at least one agent."""
        visited = set()
        for agent in self.agents:
            visited.update(agent.visited_required_areas)
        return len(visited) == len(self.required_areas)

def visualize(env, frames=500, save_video=False, video_filename='multi_agent_path_planning.mp4'):
    """
    Visualize the multi-agent path planning simulation.

    Args:
        env: The environment containing agents, obstacles, etc.
        frames: Number of frames to simulate
        save_video: Whether to save the animation as a video file
        video_filename: Filename to save the video to (if save_video is True)

    Returns:
        The animation object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
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
    required_area_patches = []
    for area in env.required_areas:
        rect = patches.Rectangle(
            (area['x'] - area['width']/2, area['y'] - area['height']/2),
            area['width'], area['height'],
            linewidth=1, edgecolor='g', facecolor='green', alpha=0.2
        )
        required_area_patches.append(ax.add_patch(rect))

    # Initialize agent artists
    agent_artists = []
    path_artists = []
    goal_artists = []
    for agent in env.agents:
        # Agent circle
        agent_circle = plt.Circle(agent.position, agent.size, color=agent.color, alpha=0.8)
        agent_artists.append(ax.add_patch(agent_circle))

        # Goal marker
        goal_marker = plt.Circle(agent.goal, 0.2, color=agent.color, alpha=0.3)
        goal_artists.append(ax.add_patch(goal_marker))

        # Path line
        path_line, = ax.plot([], [], color=agent.color, alpha=0.5, linewidth=1)
        path_artists.append(path_line)

    # Weather text and arrow
    weather_text = ax.text(env.width * 0.02, env.height * 0.97, '', fontsize=10)
    wind_arrow = ax.arrow(env.width * 0.1, env.height * 0.9, 0, 0, 
                          head_width=0.3, head_length=0.5, fc='blue', ec='blue', alpha=0.7)

    # Status text
    status_text = ax.text(env.width * 0.02, env.height * 0.93, '', fontsize=10)

    def init():
        """Initialize animation."""
        return agent_artists + path_artists + goal_artists + [weather_text, wind_arrow, status_text] + required_area_patches

    def animate(i):
        """Update animation frame."""
        env.update(0.1)  # Update with small time step

        # Update agent positions and paths
        for j, agent in enumerate(env.agents):
            agent_artists[j].center = agent.position
            if len(agent.path) > 1:
                path_artists[j].set_data([p[0] for p in agent.path], [p[1] for p in agent.path])

        # Update required area colors based on visitation
        for j, area in enumerate(env.required_areas):
            visited = False
            for agent in env.agents:
                if j in agent.visited_required_areas:
                    visited = True
                    break

            if visited:
                required_area_patches[j].set_facecolor('lightgreen')
                required_area_patches[j].set_alpha(0.4)
            else:
                required_area_patches[j].set_facecolor('green')
                required_area_patches[j].set_alpha(0.2)

        # Update weather display
        wind_speed = np.linalg.norm([env.weather['wind_x'], env.weather['wind_y']])
        wind_dir = np.arctan2(env.weather['wind_y'], env.weather['wind_x']) * 180 / np.pi
        weather_text.set_text(f'Wind: {wind_speed:.1f} m/s, Direction: {wind_dir:.0f}°, Turbulence: {env.weather["turbulence"]:.1f}')

        # Update wind arrow
        wind_arrow.set_data(x=env.width * 0.1, y=env.height * 0.9, 
                           dx=env.weather['wind_x'] * 0.5, dy=env.weather['wind_y'] * 0.5)

        # Update status text
        goals_reached = sum(1 for agent in env.agents if agent.reached_goal())
        areas_visited = len(set().union(*[agent.visited_required_areas for agent in env.agents]))
        status_text.set_text(f'Goals reached: {goals_reached}/{len(env.agents)}, Required areas visited: {areas_visited}/{len(env.required_areas)}')

        return agent_artists + path_artists + goal_artists + [weather_text, wind_arrow, status_text] + required_area_patches

    anim = FuncAnimation(fig, animate, frames=frames, init_func=init,
                         blit=True, interval=50)
    plt.title('Multi-Agent Path Planning with Continuous Space and Environmental Factors')
    plt.xlabel('X position')
    plt.ylabel('Y position')

    # Save the animation as a video file if requested
    if save_video:
        print(f"Saving animation to {video_filename}...")
        try:
            # Try to use FFMpegWriter if available
            writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(video_filename, writer=writer)
            print(f"Video saved to {video_filename}")
        except Exception as e:
            # Fallback to other writers if FFMpeg is not available
            print(f"Error using FFMpegWriter: {e}")
            try:
                # Try using PillowWriter as a fallback
                writer = animation.PillowWriter(fps=15)
                anim.save(video_filename.replace('.mp4', '.gif'), writer=writer)
                print(f"Video saved as GIF to {video_filename.replace('.mp4', '.gif')}")
            except Exception as e2:
                print(f"Error saving video: {e2}")
                print("Available writers:", animation.writers.list())

    plt.show()
    return anim

def main(save_video=False, video_filename='multi_agent_path_planning.mp4'):
    """
    Main function to set up and run the simulation.

    Args:
        save_video: Whether to save the animation as a video file
        video_filename: Filename to save the video to (if save_video is True)

    Returns:
        The environment and animation objects
    """
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
    env.add_required_area(x=3, y=13, width=2, height=2)
    env.add_required_area(x=17, y=7, width=2, height=2)

    # Add agents with different start and goal positions
    env.add_agent(Agent(0, start_pos=(2, 2), goal_pos=(18, 18), color='blue'))
    env.add_agent(Agent(1, start_pos=(18, 2), goal_pos=(2, 18), color='red'))
    env.add_agent(Agent(2, start_pos=(2, 18), goal_pos=(18, 2), color='green'))
    env.add_agent(Agent(3, start_pos=(18, 18), goal_pos=(2, 2), color='purple'))

    # Visualize the simulation
    anim = visualize(env, frames=500, save_video=save_video, video_filename=video_filename)

    return env, anim

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Agent Path Planning Simulation')
    parser.add_argument('--save-video', action='store_true', help='Save the animation as a video file')
    parser.add_argument('--video-filename', type=str, default='multi_agent_path_planning.mp4',
                        help='Filename to save the video to (if --save-video is specified)')
    args = parser.parse_args()

    # Run the simulation
    main(save_video=args.save_video, video_filename=args.video_filename)
