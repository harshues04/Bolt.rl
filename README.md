# 100m Sprint Reinforcement Learning Simulation

## Project Overview

This project implements a reinforcement learning environment that simulates a 100-meter sprint race with multiple agents competing against each other. The simulation models realistic sprint dynamics including acceleration, energy management, fatigue, and air resistance to create a competitive racing environment.

## Key Features

- **Multi-Agent Environment**: Simulates races with 4 agents (2 learning agents and 2 rule-based agents)
- **Reinforcement Learning**: Uses Independent Q-Learning (IQL) for adaptive agent behavior
- **Realistic Physics**: Models velocity, acceleration, energy consumption, and fatigue
- **Rule-Based Strategies**: Includes pre-defined "Steady" and "Explosive" racing strategies
- **Performance Analysis**: Tracks and visualizes race statistics and learning progress
- **Multiple Rendering Options**: Supports text-based (ANSI), graphical, and animated visualizations

## Technical Details

### Environment

The `SprintEnv` class is built using the OpenAI Gymnasium framework and simulates:
- A configurable race track (default 100m)
- Physical parameters like air resistance, fatigue factors, and recovery rates
- Agent capabilities with randomized variation (max speed, acceleration, stamina)
- State tracking for position, velocity, energy, and race completion

### Agent Types

1. **Q-Learning Agents (2)**: Learn optimal effort strategies through experience
   - Use discretized state spaces for position, velocity, energy, and relative position
   - Apply epsilon-greedy exploration with decaying exploration rate
   - Update Q-values through TD learning

2. **Rule-Based Agents (2)**:
   - **Steady**: Maintains consistent medium-high effort throughout race
   - **Explosive**: Starts with maximum effort, then maintains moderate effort

### Learning Process

The simulation includes:
- Training phase with configurable number of episodes
- Evaluation phase for comparing trained agents against rule-based strategies
- Progress tracking for rewards, exploration rates, and Q-table growth
- Performance metrics including win rates, average positions, and finish times

### Visualization

Multiple visualization options are provided:
- Real-time text-based race progress with colorized output
- Interactive graphical rendering of race progress
- Statistical plots for race outcomes and training metrics
- Learning curve visualization for reinforcement learning progress

## Results Analysis

After training, the system generates comprehensive statistics including:
- Win counts and podium appearances
- Average finish positions and times
- Learning progression metrics
- Agent performance comparisons

## Usage

The main function runs a complete simulation with:
1. Training phase (default 1000 episodes)
2. Evaluation phase (default 100 episodes)
3. Results visualization and statistical analysis

## Dependencies

- gymnasium: For the reinforcement learning environment
- numpy: For numerical operations
- matplotlib: For visualization and plotting
- colorama: For colored terminal output

## Applications

This simulation can be used for:
- Studying reinforcement learning in competitive multi-agent environments
- Analyzing different racing strategies for sprint events
- Educational demonstrations of agent learning and adaptation
- Testing optimization algorithms in physics-based simulations
