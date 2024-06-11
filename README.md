# Genetic Algorithm Insect Simulation

This project simulates the behavior of agents (insects) in a grid environment using a genetic algorithm. The goal of the agents is to collect resources while avoiding hazards and predators. The simulation allows for the optimization of various parameters to improve the agents' performance using a genetic algorithm. The project also includes sensitivity analysis to understand the impact of different parameters on the agents' performance.

## Introduction

In this simulation, agents move within a grid environment, searching for resources and returning them to a base. The agents can communicate with each other through "shouts" to share information about the locations of resources and the base. The environment also includes hazards and predators that the agents need to avoid. The agents' behavior is influenced by several parameters, such as their speed, detection radius, and aggressiveness. By using a genetic algorithm, we can optimize these parameters to maximize the agents' resource collection efficiency.

## Project Structure

- `GeneticAlgorithm.py`: Implements the genetic algorithm to optimize the parameters of the agents in the simulation. It uses the DEAP library for the genetic algorithm and calls functions from the `InsectSim.py` file to run the simulation and evaluate the fitness of the agents.

- `InsectSim.py`: Contains the core simulation logic, including the definition of agents, predators, resources, and hazards. This script includes functions to run the simulation and calculate the fitness of agents based on their performance. It also allows setting various parameters for the simulation.

- `Sensitivity.py`: Performs sensitivity analysis on various parameters of the simulation. It runs multiple simulations with varying parameters and visualizes the results to understand the impact of each parameter on agent performance.

- `SmoothVideo.py`: Generates a smooth video visualization of the simulation, showing the movement of agents, predators, and resources over time.

- `testing.py`: Contains testing code to verify the functionality of the simulation and genetic algorithm. It ensures that the different components of the project are working correctly.

- `simulation.mp4`: A sample video output of the simulation, demonstrating the behavior of the agents within the environment.

## Parameters

### Simulation Parameters
- **grid_size**: The size of the grid environment. Larger grids provide more space for agents to move but may also require more time to find resources.
- **num_agents**: The number of agents in the simulation. More agents can collect resources faster but may also increase competition and complexity.
- **scout_percentage**: The percentage of agents designated as scouts. Scouts search for resources but do not collect them.
- **resource_positions**: The initial positions of resources in the grid.
- **base_positions**: The positions of the base(s) where agents deliver collected resources.
- **max_hearing_distance**: The maximum distance over which agents can hear shouts from other agents.
- **predator_radius**: The radius within which predators can detect and potentially catch agents.
- **hazard_positions**: The positions of hazards in the grid.
- **hazard_radius**: The radius within which hazards can affect agents.
- **agent_speed**: The speed at which agents move within the grid.
- **base_resource_speed**: The speed at which resources and bases move (if they are movable).
- **predator_speed**: The speed at which predators move within the grid.
- **steps**: The number of steps in the simulation. More steps allow for longer simulations but require more computational time.
- **detection_radius**: The radius within which agents can detect resources and bases.
- **resource_quantity**: The initial quantity of each resource.
- **aggressiveness**: The aggressiveness level of the agents, affecting their behavior towards predators.
- **num_predators**: The number of predators in the simulation.
- **create_csv**: A boolean indicating whether to save the simulation data to a CSV file.

## Installation

To install the dependencies required for this project, you can use the provided `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```
## Usage

### Testing

To run the tests and ensure everything is working correctly, use:

```bash
python testing.py
```

### Running the Genetic Algorithm

To run the genetic algorithm and optimize the agent parameters, execute the following command:

```bash
python GeneticAlgorithm.py
```

### Performing Sensitivity Analysis

To perform sensitivity analysis on various parameters, run:

```bash
python Sensitivity.py
```

### Generating Simulation Video

To generate a video visualization of the simulation, execute:

```bash
python SmoothVideo.py
```

