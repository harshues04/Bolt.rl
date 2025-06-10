# AI Olympics: 100m Sprint Simulation

## Project Overview

AI Olympics is a reinforcement learning simulation that pits intelligent agents against each other in a virtual 100-meter sprint. The project compares how RL agents perform against traditional rule-based agents, creating an environment where different strategies can be tested and visualized.

The simulation includes:
- Two Independent Q-Learning (IQL) agents that learn optimal racing strategies
- Two rule-based agents: "Steady" (consistent pace) and "Explosive" (fast start)
- Realistic physics including fatigue, air resistance, and energy management
- Interactive visualization of races with real-time performance tracking

## Features

### Realistic Sprint Simulation
- Physically accurate modeling of sprint mechanics
- Energy management system (sprinters manage their stamina)
- Air resistance and fatigue factors
- Multi-agent environment where strategies interact

### Two Implementations
- **PyGame-based:** An interactive, visually rich simulation
- **Gymnasium Implementation:** A standardized environment for reinforcement learning research

### Reinforcement Learning
- Independent Q-Learning algorithms
- Dynamic exploration-exploitation balance
- State space representing position, velocity, energy, and track position
- Sophisticated reward function balancing speed and energy conservation

### Visualization
- Real-time race animation
- Performance charts and statistics
- Comparative analysis of different agent types

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harshues04/Bolt.rl.git
cd Bolt.rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

### PyGame Implementation
```bash
cd Pygame\ Implementation
python pygame_implementation.py
```

### Gymnasium Implementation
```bash
cd Gymnasium\ Implementation
python gym_implementation.py
```

## How It Works

### Agent Types

#### Q-Learning Agents
The reinforcement learning agents learn through experience to optimize their performance, balancing speed and energy consumption. They improve with each race through:
- Discretized state space (position, velocity, energy)
- 6 effort levels (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
- Reward function based on race completion time and position

#### Rule-Based Agents

1. **Steady Agent:** Maintains a consistent effort level (0.7), representing a methodical racing approach.

2. **Explosive Agent:** Starts with maximum effort (1.0) and then reduces to moderate effort (0.6), mimicking real sprinter strategies.

### Physics Model

The simulation uses a realistic physics model that incorporates:
- Acceleration proportional to effort and current energy
- Air resistance proportional to velocity
- Energy depletion based on effort squared
- Maximum speed based on runner capabilities

### Training Process

1. **Exploration Phase:** Agents initially explore random actions (high ε value)
2. **Exploitation Phase:** Gradually shift to using learned strategies (decreasing ε)
3. **Testing Phase:** Final evaluation with zero exploration

## Results and Analysis

After training, the system evaluates the agents' performance over 100 test episodes and produces statistics including:
- Win count and win rate
- Average finish position
- Average completion time
- Energy management efficiency

## Project Structure

```
Bolt.rl/
├── Gymnasium Implementation/   # Standardized environment for RL research
├── Pygame Implementation/      # Interactive visualization version
├── README.md                   # Project documentation
├── requirements.txt            # Required dependencies
└── LICENSE                     # MIT License
```

### Hyperparameters

You can customize the simulation by modifying these key parameters:

- `ALPHA` (0.1): Learning rate for Q-learning
- `GAMMA` (0.95): Discount factor for future rewards
- `EPSILON_START` (1.0): Initial exploration rate
- `EPSILON_DECAY` (0.995): Exploration decay rate
- `TRAINING_EPISODES` (1000): Number of training episodes
- `FATIGUE_FACTOR` (0.05): Energy consumption rate
- `RECOVERY_FACTOR` (0.01): Energy recovery rate

### State Space Discretization

Adjust the state space granularity by changing:
- `POSITION_BINS` (10): Number of position buckets
- `SPEED_BINS` (10): Number of velocity buckets
- `ENERGY_BINS` (10): Number of energy level buckets

## License

This project is licensed under the MIT License.
