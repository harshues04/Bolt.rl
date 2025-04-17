import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from colorama import init, Fore, Style
import time

# Initialize colorama for colored text output
init()

class SprintEnv(gym.Env):
    """
    A 100m sprint simulation environment built with OpenAI Gymnasium.
    Models a race with 2 Q-learning agents and 2 rule-based agents (Steady, Explosive) competing over a 100m track.
    Agents manage effort, energy, and speed to optimize performance, mimicking real-world sprint dynamics.
    
    Key Features:
    - 2 Independent Q-Learning agents learn optimal effort strategies.
    - 2 Rule-based agents follow fixed strategies: Steady (consistent pace), Explosive (fast start, then moderate).
    - Tracks position, velocity, and energy for each agent with realistic physics.
    """
    metadata = {'render_modes': ['human', 'ansi', 'graphic'], 'render_fps': 15}
    
    def __init__(self, render_mode=None, num_agents=4, track_length=100):
        super().__init__()
        
        # Environment parameters
        self.num_agents = num_agents
        self.track_length = track_length
        self.max_steps = 300
        self.step_count = 0 
        self.render_mode = render_mode
        
        # Agent parameters
        self.positions = np.zeros(self.num_agents)
        self.velocities = np.zeros(self.num_agents)
        self.energies = np.ones(self.num_agents) * 100
        self.finished = np.zeros(self.num_agents, dtype=bool)
        self.finish_times = np.ones(self.num_agents) * float('inf')
        
        # Simulation parameters
        self.time_delta = 0.1
        self.fatigue_factor = 0.05
        self.recovery_factor = 0.01
        self.air_resistance = 0.02
        self.max_acceleration = 4.0
        
        # Agent capabilities
        self.max_speeds = np.random.uniform(10.0, 12.0, size=self.num_agents)
        self.acceleration_capabilities = np.random.uniform(0.8, 1.2, size=self.num_agents)
        self.stamina_capabilities = np.random.uniform(0.8, 1.2, size=self.num_agents)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)  # Effort levels: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        
        self.observation_space = spaces.Dict({
            'position': spaces.Discrete(10),
            'velocity': spaces.Discrete(10),
            'energy': spaces.Discrete(10),
            'relative_position': spaces.Discrete(5),
        })
        
        # Storage for last step data
        self.last_observations = {}
        self.last_rewards = {}
        self.last_terminations = {}
        self.last_truncations = {}
        self.last_infos = {}
        
        # Define agent types
        self.iql_agent_ids = [0, 1]  # Two IQL agents
        self.rule_based_agent_ids = [2, 3]  # Two rule-based agents
        
        # Strategies for rule-based agents
        self.strategies = {
            "Steady": lambda pos, vel, energy: 0.7,
            "Explosive": lambda pos, vel, energy: 1.0 if pos < 0.3 else 0.6,
        }
        
        # Assign strategies
        self.agent_strategies = {2: "Steady", 3: "Explosive"}
        
        # Agent names
        self.agent_names = {
            0: "IQL Agent 1",
            1: "IQL Agent 2",
            2: "Rule-Based (Steady)",
            3: "Rule-Based (Explosive)"
        }
        
        # Performance tracking
        self.race_history = []
        
        # Graphical rendering setup
        self.fig = None
        self.ax = None
        self.anim = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state for a new race."""
        super().reset(seed=seed)
        
        self.positions = np.zeros(self.num_agents)
        self.velocities = np.zeros(self.num_agents)
        self.energies = np.ones(self.num_agents) * 100
        self.finished = np.zeros(self.num_agents, dtype=bool)
        self.finish_times = np.ones(self.num_agents) * float('inf')
        self.step_count = 0
        
        self.max_speeds = np.random.uniform(10.0, 12.0, size=self.num_agents)
        self.acceleration_capabilities = np.random.uniform(0.8, 1.2, size=self.num_agents)
        self.stamina_capabilities = np.random.uniform(0.8, 1.2, size=self.num_agents)
        
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        self.last_observations = observations
        
        return observations, {i: {} for i in range(self.num_agents)}
    
    def step(self, actions):
        """
        Advances the simulation by one time step (0.1s).
        Agents apply effort (actions), updating position, velocity, and energy based on physics and strategy.
        
        Args:
            actions: Dict of agent IDs to effort levels (0-5 for IQL, 0.0-1.0 for rule-based).
        
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Validate IQL agent actions
        for agent_id in self.iql_agent_ids:
            if agent_id not in actions:
                raise ValueError(f"Expected action for IQL agent {agent_id}, but not provided")
        
        # Generate actions for rule-based agents
        for agent_id in self.rule_based_agent_ids:
            if not self.finished[agent_id]:
                strategy_name = self.agent_strategies[agent_id]
                strategy_fn = self.strategies[strategy_name]
                pos = self.positions[agent_id] / self.track_length
                vel = self.velocities[agent_id] / 12.0
                energy = self.energies[agent_id] / 100.0
                effort = strategy_fn(pos, vel, energy)
                actions[agent_id] = effort
        
        self.step_count += 1
        
        # Update each agent's state
        for agent_id, action in actions.items():
            if not self.finished[agent_id]:
                effort = action * 0.2 if agent_id in self.iql_agent_ids else action
                effort = np.clip(effort, 0.0, 1.0)
                effective_effort = effort * min(1.0, self.energies[agent_id] / 20.0)
                acceleration = effective_effort * self.max_acceleration * self.acceleration_capabilities[agent_id]
                
                self.velocities[agent_id] += acceleration * self.time_delta
                self.velocities[agent_id] -= self.velocities[agent_id] * self.air_resistance * self.time_delta
                self.velocities[agent_id] = min(self.velocities[agent_id], self.max_speeds[agent_id])
                
                self.positions[agent_id] += self.velocities[agent_id] * self.time_delta
                
                energy_consumption = effort**2 * self.fatigue_factor * self.time_delta / self.stamina_capabilities[agent_id]
                self.energies[agent_id] = max(0.0, self.energies[agent_id] - energy_consumption)
                
                if self.positions[agent_id] >= self.track_length and not self.finished[agent_id]:
                    self.finished[agent_id] = True
                    self.finish_times[agent_id] = self.step_count * self.time_delta
        
        # Generate step outputs
        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_observation(agent_id)
            rewards[agent_id] = self._calculate_reward(agent_id)
            terminations[agent_id] = np.all(self.finished) or self.step_count >= self.max_steps
            truncations[agent_id] = False
            infos[agent_id] = {
                'position': self.positions[agent_id],
                'velocity': self.velocities[agent_id],
                'energy': self.energies[agent_id],
                'finished': self.finished[agent_id],
                'finish_time': self.finish_times[agent_id] if self.finished[agent_id] else None,
                'agent_type': self.agent_names[agent_id]
            }
        
        self.last_observations = observations
        self.last_rewards = rewards
        self.last_terminations = terminations
        self.last_truncations = truncations
        self.last_infos = infos
        
        if np.all(self.finished) or self.step_count >= self.max_steps:
            self._record_race_results()
        
        return observations, rewards, terminations, truncations, infos
    
    def _calculate_reward(self, agent_id):
        """Calculate reward based on progress and finish performance."""
        if self.finished[agent_id]:
            time_reward = max(0, 30 - self.finish_times[agent_id]) * 10
            finish_position = sum(1 for i in range(self.num_agents) 
                               if self.finished[i] and self.finish_times[i] <= self.finish_times[agent_id])
            position_reward = (self.num_agents - finish_position) * 5
            return time_reward + position_reward
        else:
            progress_reward = self.velocities[agent_id] * 0.1
            energy_reward = (self.velocities[agent_id] / max(0.1, self.energies[agent_id]/100)) * 0.05 if self.velocities[agent_id] > 0 else 0
            return progress_reward + energy_reward
    
    def _get_observation(self, agent_id):
        """Generate discretized observation for an agent."""
        pos = self.positions[agent_id] / self.track_length
        vel = self.velocities[agent_id] / 12.0
        energy = self.energies[agent_id] / 100.0
        
        pos_bin = min(9, int(pos * 10))
        vel_bin = min(9, int(vel * 10))
        energy_bin = min(9, int(energy * 10))
        
        agents_ahead = sum(1 for i in range(self.num_agents) 
                         if i != agent_id and self.positions[i] > self.positions[agent_id])
        
        if agents_ahead == 0:
            rel_pos_bin = 0
        elif agents_ahead <= self.num_agents // 3:
            rel_pos_bin = 1
        elif agents_ahead <= 2 * (self.num_agents // 3):
            rel_pos_bin = 2
        elif agents_ahead < self.num_agents - 1:
            rel_pos_bin = 3
        else:
            rel_pos_bin = 4
        
        return {'position': pos_bin, 'velocity': vel_bin, 'energy': energy_bin, 'relative_position': rel_pos_bin}
    
    def _record_race_results(self):
        """Record race outcomes for statistical analysis."""
        finish_positions = {}
        for i in range(self.num_agents):
            if self.finished[i]:
                position = sum(1 for j in range(self.num_agents) 
                            if self.finished[j] and self.finish_times[j] < self.finish_times[i]) + 1
            else:
                position = self.num_agents
            finish_positions[i] = position
        
        race_data = {
            'agent_names': {i: self.agent_names[i] for i in range(self.num_agents)},
            'positions': {i: self.positions[i] for i in range(self.num_agents)},
            'finish_times': {i: self.finish_times[i] if self.finished[i] else None for i in range(self.num_agents)},
            'finish_positions': finish_positions,
            'energy_left': {i: self.energies[i] for i in range(self.num_agents)},
            'max_velocities': {i: np.max(self.velocities[i]) for i in range(self.num_agents)}
        }
        self.race_history.append(race_data)
    
    def get_race_statistics(self):
        """Compile statistics from all recorded races."""
        if not self.race_history:
            return "No races recorded yet"
        
        stats = {i: {'agent_name': self.agent_names[i], 'races': len(self.race_history), 'wins': 0, 'podiums': 0, 
                     'avg_position': 0, 'avg_time': 0, 'finished_races': 0} for i in range(self.num_agents)}
        
        for race in self.race_history:
            for i in range(self.num_agents):
                position = race['finish_positions'][i]
                if position == 1:
                    stats[i]['wins'] += 1
                if position <= 3:
                    stats[i]['podiums'] += 1
                if race['finish_times'][i] is not None:
                    stats[i]['avg_time'] += race['finish_times'][i]
                    stats[i]['finished_races'] += 1
                stats[i]['avg_position'] += position
        
        for i in range(self.num_agents):
            if stats[i]['races'] > 0:
                stats[i]['avg_position'] /= stats[i]['races']
            if stats[i]['finished_races'] > 0:
                stats[i]['avg_time'] /= stats[i]['finished_races']
            else:
                stats[i]['avg_time'] = None
        
        return stats
    
    def render(self):
        """Render the environment based on the selected mode."""
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "graphic":
            self._render_graphic()
        else:
            return "Rendering not enabled"
    
    def _render_text(self):
        """Render the environment as a formatted text table with color highlights."""
        output = f"\nTime: {self.step_count * self.time_delta:.1f}s (Step {self.step_count})\n"
        output += f"{'Runner':<20} {'Position':<10} {'Speed':<10} {'Energy':<10} {'Status':<15}\n"
        output += "-" * 65 + "\n"
        
        for i in range(self.num_agents):
            status = "FINISHED" if self.finished[i] else "Running"
            color = Fore.GREEN if self.finished[i] else Fore.WHITE
            finish_info = f" ({self.finish_times[i]:.2f}s)" if self.finished[i] else ""
            output += f"{color}{self.agent_names[i]:<20} {self.positions[i]:<10.2f} {self.velocities[i]:<10.2f} {self.energies[i]:<10.1f} {status + finish_info:<15}{Style.RESET_ALL}\n"
        
        return output
    
    def _render_graphic(self):
        """Render the race graphically with real-time updates."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_xlim(0, self.track_length)
            self.ax.set_ylim(-1, self.num_agents)
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Runner")
            self.ax.set_title("100m Sprint Simulation")
            self.lines = [self.ax.plot([], [], 'o-', label=self.agent_names[i])[0] for i in range(self.num_agents)]
            self.ax.legend(loc='upper left')
            plt.ion()  # Interactive mode for real-time updates
        
        for i, line in enumerate(self.lines):
            line.set_data([self.positions[i]], [i])
        
        plt.draw()
        plt.pause(0.05)  # Small delay for real-time effect
    
    def plot_race_history(self, num_races=10):
        """Generate high-quality plots summarizing race statistics."""
        if not self.race_history:
            return "No races recorded yet"
        
        stats = self.get_race_statistics()
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Unique colors for agents
        
        # Wins
        wins = [stats[i]['wins'] for i in range(self.num_agents)]
        axs[0, 0].bar(range(self.num_agents), wins, color=colors)
        axs[0, 0].set_title('Wins by Agent in 100m Sprint Simulation', fontsize=12)
        axs[0, 0].set_xlabel('Agent', fontsize=10)
        axs[0, 0].set_ylabel('Number of Wins', fontsize=10)
        axs[0, 0].set_xticks(range(self.num_agents))
        axs[0, 0].set_xticklabels([self.agent_names[i] for i in range(self.num_agents)], rotation=45, ha='right')
        
        # Average Finish Positions
        avg_positions = [stats[i]['avg_position'] for i in range(self.num_agents)]
        axs[0, 1].bar(range(self.num_agents), avg_positions, color=colors)
        axs[0, 1].set_title('Average Finish Position', fontsize=12)
        axs[0, 1].set_xlabel('Agent', fontsize=10)
        axs[0, 1].set_ylabel('Position (Lower is Better)', fontsize=10)
        axs[0, 1].set_xticks(range(self.num_agents))
        axs[0, 1].set_xticklabels([self.agent_names[i] for i in range(self.num_agents)], rotation=45, ha='right')
        
        # Average Finish Times
        avg_times = [stats[i]['avg_time'] if stats[i]['avg_time'] is not None else 0 for i in range(self.num_agents)]
        axs[1, 0].bar(range(self.num_agents), avg_times, color=colors)
        axs[1, 0].set_title('Average Finish Time', fontsize=12)
        axs[1, 0].set_xlabel('Agent', fontsize=10)
        axs[1, 0].set_ylabel('Time (Seconds)', fontsize=10)
        axs[1, 0].set_xticks(range(self.num_agents))
        axs[1, 0].set_xticklabels([self.agent_names[i] for i in range(self.num_agents)], rotation=45, ha='right')
        
        # Podium Appearances
        podiums = [stats[i]['podiums'] for i in range(self.num_agents)]
        axs[1, 1].bar(range(self.num_agents), podiums, color=colors)
        axs[1, 1].set_title('Podium Appearances (Top 3)', fontsize=12)
        axs[1, 1].set_xlabel('Agent', fontsize=10)
        axs[1, 1].set_ylabel('Number of Podiums', fontsize=10)
        axs[1, 1].set_xticks(range(self.num_agents))
        axs[1, 1].set_xticklabels([self.agent_names[i] for i in range(self.num_agents)], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('race_statistics.png', dpi=300)
        plt.close(fig)
        return "Race statistics plot saved to 'race_statistics.png'"
    
    def close(self):
        """Close the environment and any open plots."""
        if self.fig is not None:
            plt.close(self.fig)


class IndependentQLearningAgent:
    """
    Q-learning agent for the 100m sprint environment.
    Learns an optimal effort strategy independently using a Q-table.
    """
    def __init__(self, agent_id, state_space, action_space, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.agent_id = agent_id
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}
        self.episode_rewards = []
        self.finish_positions = []
        self.finish_times = []
        
    def _get_state_key(self, observation):
        """Convert observation dict to a hashable state key."""
        return (observation['position'], observation['velocity'], observation['energy'], observation['relative_position'])
    
    def choose_action(self, observation):
        """Choose an action using an epsilon-greedy policy."""
        state_key = self._get_state_key(observation)
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space.n)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return np.random.choice(np.where(self.q_table[state_key] == np.max(self.q_table[state_key]))[0])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning update rule."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward if done else reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error
    
    def update_exploration_rate(self):
        """Decay exploration rate after each episode."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
    
    def record_performance(self, reward, finish_position=None, finish_time=None):
        """Record performance metrics for analysis."""
        self.episode_rewards.append(reward)
        if finish_position is not None:
            self.finish_positions.append(finish_position)
        if finish_time is not None:
            self.finish_times.append(finish_time)


def train_agents(num_episodes=1000, render_interval=200):
    """Train two IQL agents in the sprint environment."""
    env = SprintEnv(render_mode='ansi', num_agents=4)
    agents = {agent_id: IndependentQLearningAgent(agent_id=agent_id, state_space=env.observation_space, 
                                                  action_space=env.action_space, learning_rate=0.1, discount_factor=0.95, 
                                                  exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01) 
              for agent_id in env.iql_agent_ids}
    
    episode_rewards = {agent_id: [] for agent_id in agents.keys()}
    win_counts = {agent_id: 0 for agent_id in agents.keys()}
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = {agent_id: 0 for agent_id in agents.keys()}
        terminated = {agent_id: False for agent_id in agents.keys()}
        truncated = {agent_id: False for agent_id in agents.keys()}
        
        while not all(terminated.values()) and not all(truncated.values()):
            actions = {agent_id: agent.choose_action(observations[agent_id]) for agent_id, agent in agents.items()}
            next_observations, rewards, terminated, truncated, infos = env.step(actions)
            
            for agent_id, agent in agents.items():
                agent.learn(observations[agent_id], actions[agent_id], rewards[agent_id], 
                            next_observations[agent_id], terminated[agent_id] or truncated[agent_id])
                episode_reward[agent_id] += rewards[agent_id]
            
            observations = next_observations
            if episode % render_interval == 0 and episode > 0:
                print(env.render())
        
        for agent_id, agent in agents.items():
            agent.update_exploration_rate()
            episode_rewards[agent_id].append(episode_reward[agent_id])
            if env.finished[agent_id]:
                position = sum(1 for i in range(env.num_agents) 
                            if env.finished[i] and env.finish_times[i] <= env.finish_times[agent_id])
                agent.record_performance(episode_reward[agent_id], position, env.finish_times[agent_id])
                if position == 1:
                    win_counts[agent_id] += 1
        
        if episode % 500 == 0:
            print(f"Episode {episode}/{num_episodes} - Exploration Rate: {agents[0].exploration_rate:.3f}")
    
    return agents, env

def evaluate_agents(agents, env, num_episodes=100, render=True):
    """Evaluate trained agents without exploration."""
    original_exploration_rates = {agent_id: agent.exploration_rate for agent_id, agent in agents.items()}
    for agent in agents.values():
        agent.exploration_rate = 0.0
    
    win_counts = {agent_id: 0 for agent_id in agents.keys()}
    finish_positions = {agent_id: [] for agent_id in agents.keys()}
    finish_times = {agent_id: [] for agent_id in agents.keys()}
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        terminated = {agent_id: False for agent_id in agents.keys()}
        truncated = {agent_id: False for agent_id in agents.keys()}
        
        while not all(terminated.values()) and not all(truncated.values()):
            actions = {agent_id: agent.choose_action(observations[agent_id]) for agent_id, agent in agents.items()}
            next_observations, rewards, terminated, truncated, infos = env.step(actions)
            observations = next_observations
            if render and episode == num_episodes - 1:
                print(env.render() if env.render_mode == 'ansi' else '')
                env.render()  # For graphical mode
        
        for agent_id in agents.keys():
            if env.finished[agent_id]:
                position = sum(1 for i in range(env.num_agents) 
                           if env.finished[i] and env.finish_times[i] <= env.finish_times[agent_id])
                finish_positions[agent_id].append(position)
                finish_times[agent_id].append(env.finish_times[agent_id])
                if position == 1:
                    win_counts[agent_id] += 1
    
    for agent_id, rate in original_exploration_rates.items():
        agents[agent_id].exploration_rate = rate
    
    return win_counts, finish_positions, finish_times

def plot_learning_curves(agents, env):
    """Plot training performance metrics for IQL agents."""
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e']  # Colors for IQL agents
    
    # Smoothed Rewards
    plt.subplot(2, 2, 1)
    for agent_id, agent in agents.items():
        if len(agent.episode_rewards) > 0:
            window_size = 20
            smoothed_rewards = np.convolve(agent.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards, label=f"{env.agent_names[agent_id]}", color=colors[agent_id])
    plt.title('Smoothed Training Rewards', fontsize=12)
    plt.xlabel('Episodes', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.legend()
    
    # Exploration Rate Decay
    plt.subplot(2, 2, 2)
    exploration_rates = []
    rate = 1.0
    episodes = len(agents[0].episode_rewards) if agents and 0 in agents else 1000
    for _ in range(episodes):
        exploration_rates.append(rate)
        rate = max(0.01, rate * 0.995)
    plt.plot(exploration_rates, color='#2ca02c')
    plt.title('Exploration Rate Decay', fontsize=12)
    plt.xlabel('Episodes', fontsize=10)
    plt.ylabel('Exploration Rate', fontsize=10)
    
    # Average Finish Positions
    plt.subplot(2, 2, 3)
    for agent_id, agent in agents.items():
        if len(agent.finish_positions) > 0:
            window_size = min(20, len(agent.finish_positions))
            if window_size > 0:
                smoothed_positions = np.convolve(agent.finish_positions, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_positions, label=f"{env.agent_names[agent_id]}", color=colors[agent_id])
    plt.title('Average Finish Positions During Training', fontsize=12)
    plt.xlabel('Race Number', fontsize=10)
    plt.ylabel('Position (Lower is Better)', fontsize=10)
    plt.legend()
    
    # Q-table Size
    plt.subplot(2, 2, 4)
    for agent_id, agent in agents.items():
        q_table_sizes = []
        for episode in range(0, len(agent.episode_rewards), max(1, len(agent.episode_rewards) // 100)):
            if episode < len(agent.episode_rewards):
                q_table_sizes.append(len(agent.q_table))
        plt.plot(range(0, len(agent.episode_rewards), max(1, len(agent.episode_rewards) // 100))[:len(q_table_sizes)], 
                 q_table_sizes, label=f"{env.agent_names[agent_id]}", color=colors[agent_id])
    plt.title('Q-table Size (States Discovered)', fontsize=12)
    plt.xlabel('Episodes', fontsize=10)
    plt.ylabel('Number of States', fontsize=10)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    plt.close()
    return "Learning curves saved to 'learning_curves.png'"

def main():
    """Run the 100m sprint simulation with training and evaluation."""
    print("=== Starting 100m Sprint Environment Simulation ===")
    training_episodes = 1000
    evaluation_episodes = 100
    
    # Training Phase
    print(f"Training 2 IQL agents for {training_episodes} episodes...")
    agents, env = train_agents(num_episodes=training_episodes, render_interval=200)
    print("Training complete. Plotting learning curves...")
    print(plot_learning_curves(agents, env))
    
    # Evaluation Phase
    print(f"\nEvaluating agents over {evaluation_episodes} episodes with graphical rendering...")
    env = SprintEnv(render_mode='graphic', num_agents=4)
    win_counts, finish_positions, finish_times = evaluate_agents(agents, env, num_episodes=evaluation_episodes, render=True)
    print("Evaluation complete. Plotting race statistics...")
    print(env.plot_race_history())
    
    # Summary Report
    print("\n=== Final Summary ===")
    print(f"{'Agent':<20} {'Wins':<10} {'Avg Position':<15} {'Avg Time (s)':<15}")
    print("-" * 60)
    for agent_id in range(env.num_agents):
        if agent_id in agents:
            avg_pos = sum(finish_positions[agent_id]) / len(finish_positions[agent_id]) if finish_positions[agent_id] else float('inf')
            avg_time = sum(finish_times[agent_id]) / len(finish_times[agent_id]) if finish_times[agent_id] else float('inf')
            print(f"{env.agent_names[agent_id]:<20} {win_counts[agent_id]:<10} {avg_pos:<15.2f} {avg_time:<15.2f}")
        else:
            stats = env.get_race_statistics()
            avg_pos = stats[agent_id]['avg_position']
            avg_time = stats[agent_id]['avg_time'] if stats[agent_id]['avg_time'] is not None else float('inf')
            wins = stats[agent_id]['wins']
            print(f"{env.agent_names[agent_id]:<20} {wins:<10} {avg_pos:<15.2f} {avg_time:<15.2f}")
    
    print("\nSimulation concluded! See 'learning_curves.png' and 'race_statistics.png' for detailed results.")

if __name__ == "__main__":
    main()
