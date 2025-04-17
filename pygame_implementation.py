import pygame
import numpy as np
import random
import time
import os
import sys
import matplotlib
from collections import defaultdict

matplotlib.use('Agg')  # Use Agg backend for headless rendering

# Pygame Setup
WIDTH, HEIGHT = 1100, 500
LANE_HEIGHT = 60
NUM_AGENTS = 4
FINISH_LINE = 1000
FPS = 60

# Real-world scale
METERS_TO_PIXELS = 10

# Q-learning Hyperparameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
'''EPSILON_START = 1.0: Initially 100% exploration
EPSILON_END = 0.01: Minimum 1% exploration
EPSILON_DECAY = 0.995: Multiplication factor for each episode'''

# State space discretization
POSITION_BINS = 10
SPEED_BINS = 10
ENERGY_BINS = 10

# Training parameters
TRAINING_EPISODES = 1000
DISPLAY_EVERY = 200
TEST_EPISODES = 100

# Agent Types
TYPE_RL = 'RL'
TYPE_STEADY = 'Steady'
TYPE_EXPLOSIVE = 'Explosive'

# Colors
AGENT_COLORS = {
    TYPE_RL: (255, 0, 0),        # Red for RL agents
    TYPE_STEADY: (0, 0, 255),    # Blue for Steady
    TYPE_EXPLOSIVE: (0, 255, 0)  # Green for Explosive
}
LANE_COLOR = (200, 200, 200)
TRACK_COLOR = (128, 128, 128)
GRASS_COLOR = (34, 139, 34)

# Initialize Q-Tables - use sparse representation with defaultdict
def create_q_tables(num_rl_agents):
    return [defaultdict(float) for _ in range(num_rl_agents)]

Q_TABLES = create_q_tables(2)  # 2 RL agents

class Runner:
    def __init__(self, x, y, agent_id, agent_type):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 40
        self.agent_type = agent_type
        self.color = AGENT_COLORS.get(agent_type, (255, 0, 0))
        self.speed = 0.0
        self.max_speed = random.uniform(10.0, 12.0)
        self.min_speed = 0.0
        self.energy = 100.0
        self.finished = False
        self.agent_id = agent_id
        self.finish_time = None
        self.frames_elapsed = 0        
        self.animation_frame = 0
        self.animation_counter = 0
        self.distance_covered = 0
        self.max_acceleration = 4.0
        self.fatigue_factor = 0.05
        self.air_resistance = 0.02
        self.energy_ratio = 1.0

    def get_state(self):
        position_bin = min(POSITION_BINS - 1, int(self.x / FINISH_LINE * POSITION_BINS))
        speed_bin = min(SPEED_BINS - 1, int(self.speed / self.max_speed * SPEED_BINS))
        energy_bin = min(ENERGY_BINS - 1, int(self.energy / 100.0 * ENERGY_BINS))
        return (position_bin, speed_bin, energy_bin)

    def choose_action(self, epsilon):
        raise NotImplementedError("Subclasses must implement choose_action")

    def move(self, action):
        if self.finished:
            return
        
        self.frames_elapsed += 1  # Increment frame counter
        self.energy_ratio = min(1.0, self.energy / 20.0)
        
        if self.agent_type == TYPE_RL:
            effort = action * 0.2  # RL uses discrete 0-5
        else:
            effort = action
            
        effort = np.clip(effort, 0.0, 1.0)
        effective_effort = effort * self.energy_ratio
        acceleration = effective_effort * self.max_acceleration
        
        drag = self.speed * self.air_resistance
        net_acceleration = acceleration - drag
        
        self.speed += net_acceleration / FPS
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))
        
        pixels_per_frame = self.speed * METERS_TO_PIXELS / FPS
        self.x += pixels_per_frame
        self.distance_covered += pixels_per_frame / METERS_TO_PIXELS
        
        energy_consumption = effort**2 * self.fatigue_factor / FPS
        self.energy = max(0.0, self.energy - energy_consumption)
        
        if self.speed > 1.0:
            self.animation_counter += self.speed / 2
            if self.animation_counter >= 5:
                self.animation_frame = (self.animation_frame + 1) % 3
                self.animation_counter = 0
        
        if self.x >= FINISH_LINE and not self.finished:
            self.finished = True
            self.finish_time = self.frames_elapsed / FPS  # Calculate time based on frames

    def update_q_table(self, action, reward, next_state):
        pass

class RLRunner(Runner):
    def __init__(self, x, y, agent_id):
        super().__init__(x, y, agent_id, TYPE_RL)
    
    def choose_action(self, epsilon):
        state = self.get_state()
        if np.random.rand() < epsilon:
            return random.randint(0, 5)
        
        q_values = np.zeros(6)
        for a in range(6):
            q_values[a] = Q_TABLES[self.agent_id].get((state, a), 0.0)
        
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        return np.random.choice(max_actions)
    
    def update_q_table(self, action, reward, next_state):
        current_state = self.get_state()
        max_future_q = 0.0
        for a in range(6):
            max_future_q = max(max_future_q, Q_TABLES[self.agent_id].get((next_state, a), 0.0))
        
        current_q = Q_TABLES[self.agent_id].get((current_state, action), 0.0)
        new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
        Q_TABLES[self.agent_id][(current_state, action)] = new_q

class SteadyRunner(Runner):
    def __init__(self, x, y, agent_id):
        super().__init__(x, y, agent_id, TYPE_STEADY)
    
    def choose_action(self, epsilon):
        return 0.7

class ExplosiveRunner(Runner):
    def __init__(self, x, y, agent_id):
        super().__init__(x, y, agent_id, TYPE_EXPLOSIVE)
    
    def choose_action(self, epsilon):
        distance = self.distance_covered / 100.0
        return 1.0 if distance < 0.3 else 0.6

class RenderManager:
    def __init__(self):
        self.initialized = False
        self.screen = None
        self.font = None
        self.small_font = None
        self.sprite_image = None
        self.text_cache = {}

    def initialize(self):
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("100m Dash - RL vs Rule-Based Simulation")
            self.font = pygame.font.SysFont(None, 24)
            self.small_font = pygame.font.SysFont(None, 18)
            self.load_sprite()
            self.initialized = True

    def load_sprite(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            sprite_path = os.path.join(base_path, "runners.gif")
            sprite = pygame.image.load(sprite_path).convert_alpha()
            self.sprite_image = pygame.transform.scale(sprite, (40, 40))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Error loading sprite: {e}. Using default rectangles.")
            self.sprite_image = None

    def get_text_surface(self, text, color=(0, 0, 0), small=False):
        key = (text, color, small)
        if key not in self.text_cache:
            font = self.small_font if small else self.font
            self.text_cache[key] = font.render(text, True, color)
        return self.text_cache[key]

    def clear_text_cache(self):
        if len(self.text_cache) > 100:
            self.text_cache.clear()

    def render_track(self):
        self.screen.fill(GRASS_COLOR)
        track_height = NUM_AGENTS * LANE_HEIGHT
        pygame.draw.rect(self.screen, TRACK_COLOR, (0, 0, WIDTH, track_height))
        
        for i in range(NUM_AGENTS + 1):
            pygame.draw.line(self.screen, LANE_COLOR, (0, i * LANE_HEIGHT), (WIDTH, i * LANE_HEIGHT), 2)
        
        for i in range(1, 11):
            x_pos = i * 100 * METERS_TO_PIXELS / 10
            pygame.draw.line(self.screen, (255, 255, 255), (x_pos, 0), (x_pos, track_height), 1)
            if i < 10:
                distance_text = self.get_text_surface(f"{i * 10}m", (255, 255, 255))
                self.screen.blit(distance_text, (x_pos + 2, track_height + 5))
        
        pygame.draw.line(self.screen, (255, 255, 255), (FINISH_LINE, 0), (FINISH_LINE, track_height), 3)
        finish_text = self.get_text_surface("FINISH", (255, 255, 255))
        self.screen.blit(finish_text, (FINISH_LINE - 30, track_height + 5))
        
        for i in range(NUM_AGENTS):
            lane_text = self.get_text_surface(f"{i + 1}", (255, 255, 255))
            self.screen.blit(lane_text, (10, i * LANE_HEIGHT + 15))

    def render_runner(self, runner):
        y_offset = -2 if runner.animation_frame == 1 else 2 if runner.animation_frame == 2 else 0
        
        if self.sprite_image:
            self.screen.blit(self.sprite_image, (runner.x, runner.y + y_offset))
        else:
            pygame.draw.rect(self.screen, runner.color, (runner.x, runner.y + y_offset, runner.width, runner.height))
            pygame.draw.rect(self.screen, (0, 0, 0), (runner.x, runner.y + y_offset, runner.width, runner.height), 1)
        
        speed_bar_width = 40
        speed_bar_height = 6
        speed_ratio = runner.speed / runner.max_speed
        pygame.draw.rect(self.screen, (0, 0, 0), (runner.x, runner.y - 10, speed_bar_width, speed_bar_height), 1)
        pygame.draw.rect(self.screen, (0, 255, 0), (runner.x, runner.y - 10, int(speed_bar_width * speed_ratio), speed_bar_height))
        
        type_text = self.get_text_surface(f"{runner.agent_type} ({runner.energy:.1f})", (0, 0, 0), small=True)
        self.screen.blit(type_text, (runner.x, runner.y - 26))

    def render_info_panel(self, episode, total_episodes, epsilon, race_time, phase="Training"):
        track_height = NUM_AGENTS * LANE_HEIGHT
        info_panel_y = track_height + 30
        
        phase_text = self.get_text_surface(f"Phase: {phase}", (0, 0, 0))
        self.screen.blit(phase_text, (20, info_panel_y - 25))
        
        episode_text = self.get_text_surface(f"Episode: {episode}/{total_episodes} | ε: {epsilon:.2f}", (0, 0, 0))
        self.screen.blit(episode_text, (20, info_panel_y))
        
        time_text = self.get_text_surface(f"Race Time: {race_time:.2f}s", (0, 0, 0))
        self.screen.blit(time_text, (20, info_panel_y + 25))
        
        pygame.display.flip()

class RaceEnvironment:
    def __init__(self):
        self.runners = []
        self.agent_names = {
            0: "IQL Agent 1",
            1: "IQL Agent 2",
            2: "Rule-Based (Steady)",
            3: "Rule-Based (Explosive)"
        }
        self.race_finished = False
        self.frames_elapsed = 0
        self.finish_positions = {i: [] for i in range(NUM_AGENTS)}
        self.finish_times = {i: [] for i in range(NUM_AGENTS)}
        self.wins = {i: 0 for i in range(NUM_AGENTS)}
        self.current_finished_count = 0
        self.test_finish_positions = {i: [] for i in range(NUM_AGENTS)}
        self.test_finish_times = {i: [] for i in range(NUM_AGENTS)}
        self.test_wins = {i: 0 for i in range(NUM_AGENTS)}

    def reset(self):
        self.runners = [
            RLRunner(50, 0.5 * LANE_HEIGHT - 20, 0),
            RLRunner(50, 1.5 * LANE_HEIGHT - 20, 1),
            SteadyRunner(50, 2.5 * LANE_HEIGHT - 20, 2),
            ExplosiveRunner(50, 3.5 * LANE_HEIGHT - 20, 3)
        ]
        self.race_finished = False
        self.frames_elapsed = 0
        self.current_finished_count = 0

    def calculate_reward(self, runner):
        if runner.finished:
            time_reward = max(0, 30 - runner.finish_time) * 10
            position = self.current_finished_count
            position_reward = (NUM_AGENTS - position) * 5
            return time_reward + position_reward
        else:
            speed_component = runner.speed * 0.1
            energy_efficiency = speed_component * (runner.energy / 100.0) * 0.05 if runner.speed > 0 else 0
            return speed_component + energy_efficiency
    '''
    Reward function:
    Two-Part Structure: Small continuous rewards during the race for speed and energy management, plus large terminal rewards for race completion and placement.
    Time & Position Rewards: When finishing, runners earn up to 300 points for fast times (10 points per second under 30s) and up to 15 points for placing higher than opponents.
    Speed Component: During the race, runners earn small rewards (speed × 0.1) for maintaining higher speeds, encouraging forward movement.
    Energy Efficiency: A smaller reward factor that diminishes as energy depletes, encouraging strategic energy conservation rather than all-out sprinting.
    Strategic Balance: The reward function forces agents to optimize multiple competing objectives - finishing quickly, beating opponents, and managing energy efficiently throughout the race.'''
    def step(self, epsilon):
        if self.race_finished:
            return True
        
        self.frames_elapsed += 1  # Increment frame counter
            
        if all(runner.finished for runner in self.runners):
            self._process_race_results(is_test=False)  # During training, pass is_test=False
            return True
            
        for runner in self.runners:
            if runner.finished:
                continue
                
            action = runner.choose_action(epsilon)
            old_state = runner.get_state()
            runner.move(action)
            new_state = runner.get_state()
            
            if runner.finished and not runner.finish_time:
                self.current_finished_count += 1
                
            reward = self.calculate_reward(runner)
            runner.update_q_table(action, reward, new_state)
            
        return False

    def _process_race_results(self, is_test=False):
        self.race_finished = True
        finished_runners = sorted([r for r in self.runners if r.finished], key=lambda x: x.finish_time or float('inf'))
        
        # Add a DNF (did not finish) time penalty for runners who didn't finish
        max_time = self.frames_elapsed / FPS + 5.0  # DNF penalty: add 5 seconds to max time
        
        for i, runner in enumerate(self.runners):
            position = None
            time = None
            
            if runner.finished:
                # Find this runner in the finished list to get position
                for pos, finished_runner in enumerate(finished_runners, 1):
                    if finished_runner.agent_id == runner.agent_id:
                        position = pos
                        time = runner.finish_time
                        break
            else:
                # For runners who didn't finish, assign a position after all finishers
                # and use the max time with penalty
                position = len(finished_runners) + 1
                time = max_time
                
            if is_test:
                self.test_finish_positions[runner.agent_id].append(position)
                self.test_finish_times[runner.agent_id].append(time)
                if position == 1:
                    self.test_wins[runner.agent_id] += 1
            else:
                self.finish_positions[runner.agent_id].append(position)
                self.finish_times[runner.agent_id].append(time)
                if position == 1:
                    self.wins[runner.agent_id] += 1

    def get_race_time(self):
        return self.frames_elapsed / FPS

    def print_summary(self, use_test_data=False):
        if use_test_data:
            print("\n=== Test Episodes Summary (100 episodes) ===")
            wins = self.test_wins
            positions = self.test_finish_positions
            times = self.test_finish_times
        else:
            print("\n=== Training Summary ===")
            wins = self.wins
            positions = self.finish_positions
            times = self.finish_times
            
        print(f"{'Agent':<20} {'Wins':<10} {'Avg Position':<15} {'Avg Time (s)':<15}")
        print("-" * 60)
        for i in range(NUM_AGENTS):
            avg_pos = sum(positions[i]) / len(positions[i]) if positions[i] else float('inf')
            avg_time = sum(times[i]) / len(times[i]) if times[i] else float('inf')
            print(f"{self.agent_names[i]:<20} {wins[i]:<10} {avg_pos:<15.2f} {avg_time:<15.2f}")

class RaceGame:
    def __init__(self):
        self.renderer = RenderManager()
        self.environment = RaceEnvironment()
        self.running = True
        self.episode = 0
        self.epsilon = EPSILON_START
        self.clock = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
        return True

    def should_render_episode(self):
        return self.episode % DISPLAY_EVERY == 0 or self.episode > TRAINING_EPISODES

    def run(self, episodes=TRAINING_EPISODES):
        # Training phase
        self.episode = 0
        while self.episode < episodes and self.running:
            self.epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** self.episode))
            self.environment.reset()
            self.episode += 1
            
            should_render = self.should_render_episode()
            if should_render and not self.renderer.initialized:
                self.renderer.initialize()
                self.clock = pygame.time.Clock()
            
            progress = self.episode / episodes
            print(f'\rEpisode {self.episode}/{episodes} | Progress: {progress:.1%} (ε={self.epsilon:.2f})', end='')
            
            frame_count = 0
            max_frames = 60 * 30
            
            while self.running and not self.environment.race_finished and frame_count < max_frames:
                if should_render and not self.handle_events():
                    pygame.quit()
                    return
                
                self.environment.step(self.epsilon)
                
                if should_render:
                    self.renderer.render_track()
                    for runner in self.environment.runners:
                        self.renderer.render_runner(runner)
                    self.renderer.render_info_panel(
                        self.episode,
                        episodes,
                        self.epsilon,
                        self.environment.get_race_time()
                    )
                    self.clock.tick(FPS)
                    
                    if frame_count % 100 == 0:
                        self.renderer.clear_text_cache()
                
                frame_count += 1
            
            if self.episode % 500 == 0:
                print(f"\nEpisode {self.episode}/{episodes} - Exploration Rate: {self.epsilon:.3f}")
        
        self.environment.print_summary(use_test_data=False)
        
        print("\nTraining complete! Running 100 test episodes with zero exploration...")
        
        self.epsilon = 0.0
        if not self.renderer.initialized:
            self.renderer.initialize()
            self.clock = pygame.time.Clock()
            
        for test_ep in range(TEST_EPISODES):
            test_episode_num = test_ep + 1
            self.environment.reset()
            
            if test_episode_num % 10 == 0:
                print(f'\rTest Episode {test_episode_num}/{TEST_EPISODES} | Progress: {test_episode_num/TEST_EPISODES:.1%}', end='')
            
            should_render = test_episode_num % 20 == 0 or test_episode_num == TEST_EPISODES
            
            frame_count = 0
            max_frames = 60 * 30
            
            while not self.environment.race_finished and frame_count < max_frames and self.running:
                if should_render and not self.handle_events():
                    pygame.quit()
                    return
                    
                self.environment.step(0.0)
                
                if should_render:
                    self.renderer.render_track()
                    for runner in self.environment.runners:
                        self.renderer.render_runner(runner)
                    self.renderer.render_info_panel(
                        test_episode_num,
                        TEST_EPISODES,
                        0.0,
                        self.environment.get_race_time(),
                        phase="Testing"
                    )
                    self.clock.tick(FPS)
                frame_count += 1
            
            if not self.environment.race_finished:
                self.environment.race_finished = True  # Ensure race is marked finished
            self.environment._process_race_results(is_test=True)  # Always process test results
            
            if should_render:
                time.sleep(0.5)
        
        self.environment.print_summary(use_test_data=True)
        pygame.quit()

if __name__ == "__main__":
    game = RaceGame()
    game.run()