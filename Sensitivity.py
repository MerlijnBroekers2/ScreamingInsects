import numpy as np
import matplotlib.pyplot as plt
from InsectSim import run_simulation
import seaborn as sns
import time
import pandas as pd

def average_fitness_over_runs(runs, **simulation_params):
    fitness_values = []
    for _ in range(runs):
        fitness, _ = run_simulation(**simulation_params)
        fitness_values.append(fitness)
    return np.mean(fitness_values)


def grid_size_vs_num_agents():
    grid_sizes = np.arange(20, 201, 40)
    num_agents = np.arange(20, 401, 40)
    fitness_matrix = np.zeros((len(grid_sizes), len(num_agents)))

    for i, grid_size in enumerate(grid_sizes):
        for j, agents in enumerate(num_agents):
            fitness_matrix[i, j] = average_fitness_over_runs(
                5, grid_size=grid_size, num_agents=agents, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[20, 400, 200, 20])
    plt.colorbar(label='Average Fitness')
    plt.xlabel('Number of Agents')
    plt.ylabel('Grid Size')
    plt.title('Grid Size vs Number of Agents')
    plt.show()


def base_resource_speed_vs_fitness():
    base_resource_speeds = np.arange(0.05, 0.55, 0.05)
    fitness_values = []

    for speed in base_resource_speeds:
        fitness = average_fitness_over_runs(
            10, grid_size=80, num_agents=100, scout_percentage=0.1,
            resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
            max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=speed, predator_speed=0.25,
            steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            safe_zone=None, create_csv=False)
        fitness_values.append(fitness)

    plt.figure()
    plt.plot(base_resource_speeds, fitness_values, marker='o')
    plt.xlabel('Base Resource Speed')
    plt.ylabel('Average Fitness')
    plt.title('Base Resource Speed vs Fitness')
    plt.grid(True)
    plt.show()


def predator_speed_vs_fitness():
    predator_speeds = np.arange(0.1, 1.1, 0.1)
    fitness_values = []

    for speed in predator_speeds:
        fitness = average_fitness_over_runs(
            10, grid_size=80, num_agents=100, scout_percentage=0.1,
            resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
            max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=speed,
            steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            safe_zone=None, create_csv=False)
        fitness_values.append(fitness)

    plt.figure()
    plt.plot(predator_speeds, fitness_values, marker='o')
    plt.xlabel('Predator Speed')
    plt.ylabel('Average Fitness')
    plt.title('Predator Speed vs Fitness')
    plt.grid(True)
    plt.show()


def detection_radius_vs_fitness():
    detection_radii = np.arange(1, 21, 1)
    fitness_values = []

    for radius in detection_radii:
        fitness = average_fitness_over_runs(
            10, grid_size=80, num_agents=100, scout_percentage=0.1,
            resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
            max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
            steps=500, detection_radius=radius, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            safe_zone=None, create_csv=False)
        fitness_values.append(fitness)

    plt.figure()
    plt.plot(detection_radii, fitness_values, marker='o')
    plt.xlabel('Detection Radius')
    plt.ylabel('Average Fitness')
    plt.title('Detection Radius vs Fitness')
    plt.grid(True)
    plt.show()


def fitness_vs_steps():
    steps_values = np.arange(100, 1100, 100)
    fitness_values = []

    for steps in steps_values:
        fitness = average_fitness_over_runs(
            10, grid_size=80, num_agents=100, scout_percentage=0.1,
            resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
            max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
            steps=steps, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            safe_zone=None, create_csv=False)
        fitness_values.append(fitness)

    plt.figure()
    plt.plot(steps_values, fitness_values, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Average Fitness')
    plt.title('Fitness vs Steps of the Simulation')
    plt.grid(True)
    plt.show()


def grid_size_vs_detection_radius():
    grid_sizes = np.arange(40, 201, 20)
    detection_radii = np.arange(1, 21, 1)
    fitness_matrix = np.zeros((len(grid_sizes), len(detection_radii)))

    for i, grid_size in enumerate(grid_sizes):
        for j, radius in enumerate(detection_radii):
            fitness_matrix[i, j] = average_fitness_over_runs(
                10, grid_size=grid_size, num_agents=100, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=500, detection_radius=radius, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[1, 20, 200, 40])
    plt.colorbar(label='Average Fitness')
    plt.xlabel('Detection Radius')
    plt.ylabel('Grid Size')
    plt.title('Grid Size vs Detection Radius')
    plt.show()


def num_agents_vs_agent_speed():
    num_agents = np.arange(20, 401, 20)
    agent_speeds = np.arange(0.2, 2.2, 0.2)
    fitness_matrix = np.zeros((len(num_agents), len(agent_speeds)))

    for i, agents in enumerate(num_agents):
        for j, speed in enumerate(agent_speeds):
            fitness_matrix[i, j] = average_fitness_over_runs(
                10, grid_size=80, num_agents=agents, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=7, agent_speed=speed, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[0.2, 2.0, 400, 20])
    plt.colorbar(label='Average Fitness')
    plt.xlabel('Agent Speed')
    plt.ylabel('Number of Agents')
    plt.title('Number of Agents vs Agent Speed')
    plt.show()


def predator_speed_vs_hazard_radius():
    predator_speeds = np.arange(0.1, 1.1, 0.1)
    hazard_radii = np.arange(1, 11, 1)
    fitness_matrix = np.zeros((len(predator_speeds), len(hazard_radii)))

    for i, speed in enumerate(predator_speeds):
        for j, radius in enumerate(hazard_radii):
            fitness_matrix[i, j] = average_fitness_over_runs(
                10, grid_size=80, num_agents=100, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=radius, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=speed,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[1, 10, 1.0, 0.1])
    plt.colorbar(label='Average Fitness')
    plt.xlabel('Hazard Radius')
    plt.ylabel('Predator Speed')
    plt.title('Predator Speed vs Hazard Radius')
    plt.show()


def grid_size_vs_num_agents_3d():
    grid_sizes = np.arange(40, 201, 20)
    num_agents = np.arange(20, 401, 20)
    X, Y = np.meshgrid(grid_sizes, num_agents)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = average_fitness_over_runs(
                10, grid_size=grid_sizes[j], num_agents=num_agents[i], scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Number of Agents')
    ax.set_zlabel('Average Fitness')
    ax.set_title('Grid Size and Number of Agents vs Fitness')
    plt.show()

def fitness_with_randomization(runs, **simulation_params):
    fitness_values = []
    for seed in range(runs):
        np.random.seed(seed)
        fitness, _ = run_simulation(**simulation_params)
        fitness_values.append(fitness)
    return fitness_values

def analyze_randomization_impact():
    grid_size = 80
    num_agents = 100
    scout_percentage = 0.1
    resource_positions = [(10, 10), (70, 60)]
    base_positions = [(50, 50)]
    max_hearing_distance = 10
    predator_radius = 5
    hazard_positions = [(25, 25), (60, 60)]
    hazard_radius = 7
    agent_speed = 1.0
    base_speed = 0.0
    resource_speed = 0.0
    predator_speed = 0.25
    steps = 500
    detection_radius = 5
    resource_quantity = 300
    aggressiveness = 0.5
    num_predators = 2

    # Run the simulation with randomization
    runs = 30
    fitness_values = fitness_with_randomization(
        runs, grid_size=grid_size, num_agents=num_agents, scout_percentage=scout_percentage,
        resource_positions=resource_positions, base_positions=base_positions,
        max_hearing_distance=max_hearing_distance, predator_radius=predator_radius,
        hazard_positions=hazard_positions, hazard_radius=hazard_radius,
        agent_speed=agent_speed, base_speed=base_speed, resource_speed=resource_speed, predator_speed=predator_speed,
        steps=steps, detection_radius=detection_radius, resource_quantity=resource_quantity,
        aggressiveness=aggressiveness, num_predators=num_predators, safe_zone=None, create_csv=False)

    # Plot the distribution of fitness values
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fitness_values)
    plt.title('Impact of Randomization on Fitness')
    plt.ylabel('Fitness')
    plt.xlabel('Simulation Runs')
    plt.show()


def predator_radius_vs_aggressiveness():
    predator_radii = np.arange(1, 11, 1)
    aggressiveness_levels = np.arange(0.1, 1.1, 0.1)
    fitness_matrix = np.zeros((len(predator_radii), len(aggressiveness_levels)))

    for i, radius in enumerate(predator_radii):
        for j, aggressiveness in enumerate(aggressiveness_levels):
            fitness_matrix[i, j] = average_fitness_over_runs(
                40, grid_size=100, num_agents=50, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 70)], base_positions=[(50, 50)],
                max_hearing_distance=45, predator_radius=radius, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=1, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=250, detection_radius=5, resource_quantity=300, aggressiveness=aggressiveness, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[0.1, 1.0, 1, 10])
    plt.colorbar(label='Average Fitness')
    plt.xlabel('Aggressiveness')
    plt.ylabel('Predator Radius')
    plt.title('Predator Radius vs Aggressiveness')
    plt.show()


def measure_runtime(num_agents, steps, **simulation_params):
    start_time = time.time()
    run_simulation(num_agents=num_agents, steps=steps, **simulation_params)
    return time.time() - start_time

def sensitivity_analysis_runtime():
    num_agents = np.arange(20, 401, 40)
    steps = np.arange(100, 2001, 200)
    runtime_matrix = np.zeros((len(num_agents), len(steps)))

    for i, agents in enumerate(num_agents):
        for j, step in enumerate(steps):
            runtime_matrix[i, j] = measure_runtime(
                num_agents=agents, steps=step, grid_size=80, scout_percentage=0.1,
                resource_positions=[(10, 10), (70, 60)], base_positions=[(50, 50)],
                max_hearing_distance=10, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
                hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                safe_zone=None, create_csv=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(runtime_matrix, xticklabels=steps, yticklabels=num_agents, cmap='viridis', annot=True, fmt=".2f")
    plt.xlabel('Steps')
    plt.ylabel('Number of Agents')
    plt.title('Runtime Sensitivity Analysis\n(Number of Agents vs Steps)')
    plt.show()


def is_valid_position(position, other_positions, grid_size, buffer=5):
    """Check if a position is valid, i.e., within the grid and not overlapping with other positions."""
    x, y = position
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    for other in other_positions:
        ox, oy = other
        if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < buffer:
            return False
    return True


def generate_valid_positions(grid_size, num_positions, buffer=5):
    """Generate valid positions for resources and bases."""
    positions = []
    while len(positions) < num_positions:
        candidate = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        if is_valid_position(candidate, positions, grid_size, buffer):
            positions.append(candidate)
    return positions


def agents_hearing_gridsize():
    # Define the ranges for grid size, number of agents, and hearing distance
    grid_sizes = np.arange(40, 201, 40)  # Grid sizes from 40 to 200 in steps of 40
    num_agents = np.arange(20, 101, 20)  # Number of agents from 20 to 100 in steps of 20
    hearing_distances = np.arange(20, 101, 20)  # Hearing distances from 20 to 100 in steps of 20

    # Initialize a dictionary to store fitness values
    results = []

    # Iterate over all combinations of the parameters
    for grid_size in grid_sizes:
        resource_positions = generate_valid_positions(grid_size, 2, buffer=5)
        base_positions = generate_valid_positions(grid_size, 1, buffer=5)
        hazard_positions = generate_valid_positions(grid_size, 2, buffer=5)

        for agents in num_agents:
            for hearing_distance in hearing_distances:
                average_fitness = average_fitness_over_runs(
                    10, grid_size=grid_size, num_agents=agents, scout_percentage=0.1,
                    resource_positions=resource_positions, base_positions=base_positions,
                    max_hearing_distance=hearing_distance, predator_radius=5, hazard_positions=hazard_positions,
                    hazard_radius=1, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                    steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=0,
                    safe_zone=None, create_csv=False)
                results.append((grid_size, agents, hearing_distance, average_fitness))

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=['grid_size', 'num_agents', 'hearing_distance', 'fitness'])

    # Create plots to visualize the sensitivity analysis
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.heatmap(
        data=df_results.pivot_table(index='grid_size', columns='num_agents', values='fitness'),
        ax=axes[0],
        cmap='viridis',
        cbar_kws={'label': 'Average Fitness'},
    )
    axes[0].set_title('Grid Size vs Number of Agents')
    axes[0].set_xlabel('Number of Agents')
    axes[0].set_ylabel('Grid Size')

    sns.heatmap(
        data=df_results.pivot_table(index='grid_size', columns='hearing_distance', values='fitness'),
        ax=axes[1],
        cmap='viridis',
        cbar_kws={'label': 'Average Fitness'},
    )
    axes[1].set_title('Grid Size vs Hearing Distance')
    axes[1].set_xlabel('Hearing Distance')
    axes[1].set_ylabel('Grid Size')

    sns.heatmap(
        data=df_results.pivot_table(index='num_agents', columns='hearing_distance', values='fitness'),
        ax=axes[2],
        cmap='viridis',
        cbar_kws={'label': 'Average Fitness'},
    )
    axes[2].set_title('Number of Agents vs Hearing Distance')
    axes[2].set_xlabel('Hearing Distance')
    axes[2].set_ylabel('Number of Agents')

    plt.tight_layout()
    plt.show()

def fitness_vs_hearing_distance():
    # Define the grid size and number of agents
    grid_size = 100
    num_agents = 50
    hearing_distances = np.arange(5, 101, 5)  # Hearing distances from 1 to 100 in steps of 5

    # Initialize a list to store fitness values
    fitness_values = []

    # Iterate over hearing distances and calculate average fitness
    for hearing_distance in hearing_distances:
        average_fitness = average_fitness_over_runs(
            40, grid_size=grid_size, num_agents=num_agents, scout_percentage=0.1,
            resource_positions=[(10, 10), (90, 90)], base_positions=[(50, 50)],
            max_hearing_distance=hearing_distance, predator_radius=5, hazard_positions=[(25, 25), (75, 75)],
            hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
            steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=0,
            safe_zone=None, create_csv=False)
        fitness_values.append(average_fitness)

    # Plot fitness against hearing distance
    plt.figure(figsize=(10, 6))
    plt.plot(hearing_distances, fitness_values, marker='o')
    plt.xlabel('Hearing Distance')
    plt.ylabel('Average Fitness')
    plt.title('Fitness vs Hearing Distance for Grid Size 100 and 50 Agents')
    plt.grid(True)
    plt.show()

def fitness_vs_hearing_distance_and_agents():
    # Define the grid size and the range for hearing distances and number of agents
    grid_size = 100
    hearing_distances = np.arange(5, 101, 5)  # Hearing distances from 5 to 100 in steps of 5
    num_agents_list = np.arange(50, 101, 25)  # Number of agents from 20 to 100 in steps of 20

    # Initialize a dictionary to store results
    results = {num_agents: [] for num_agents in num_agents_list}

    # Iterate over number of agents and hearing distances, and calculate average fitness
    for num_agents in num_agents_list:
        for hearing_distance in hearing_distances:
            average_fitness = average_fitness_over_runs(
                40, grid_size=grid_size, num_agents=num_agents, scout_percentage=0.1,
                resource_positions=[(10, 10), (90, 90)], base_positions=[(50, 50)],
                max_hearing_distance=hearing_distance, predator_radius=5, hazard_positions=[(25, 25), (75, 75)],
                hazard_radius=7, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=0,
                safe_zone=None, create_csv=False)
            results[num_agents].append(average_fitness)

    # Plot fitness against hearing distance for different number of agents
    plt.figure(figsize=(12, 8))
    for num_agents, fitness_values in results.items():
        plt.plot(hearing_distances, fitness_values, marker='o', label=f'{num_agents} Agents')

    plt.xlabel('Hearing Distance')
    plt.ylabel('Average Fitness')
    plt.title('Fitness vs Hearing Distance for Different Numbers of Agents')
    plt.legend()
    plt.grid(True)
    plt.show()


def fitness_with_spread(runs, **simulation_params):
    fitness_values = []
    for _ in range(runs):
        fitness, _ = run_simulation(**simulation_params)
        fitness_values.append(fitness)
    return fitness_values


def fitness_vs_steps_with_spread():
    steps_values = np.arange(10, 101, 1)
    all_fitness_values = []

    for steps in steps_values:
        fitness_values = fitness_with_spread(
            50, grid_size=100, num_agents=50, scout_percentage=0.1,
            resource_positions=[(10, 10), (70, 70)], base_positions=[(50, 50)],
            max_hearing_distance=45, predator_radius=5, hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=1, agent_speed=1.0, base_speed=0.0, resource_speed=0.0, predator_speed=0.25,
            steps=steps, detection_radius=5, resource_quantity=4000, aggressiveness=0.5, num_predators=0,
            safe_zone=None, create_csv=False)
        all_fitness_values.append((steps, fitness_values))

    # Calculate mean, 25th, and 75th percentiles
    steps_list = []
    mean_fitness = []
    lower_percentile = []
    upper_percentile = []

    for steps, fitness_values in all_fitness_values:
        steps_list.append(steps)
        mean_fitness.append(np.mean(fitness_values))
        lower_percentile.append(np.percentile(fitness_values, 25))
        upper_percentile.append(np.percentile(fitness_values, 75))

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(steps_list, mean_fitness, label='Mean Fitness', color='b')
    plt.fill_between(steps_list, lower_percentile, upper_percentile, color='gray', alpha=0.3,
                     label='25th-75th Percentile')

    plt.title('Sensitivity Analysis: Fitness vs Steps')
    plt.xlabel('Steps')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # fitness_vs_hearing_distance()
    fitness_vs_steps_with_spread()
    # grid_size_vs_num_agents()
    # predator_radius_vs_aggressiveness()
    # sensitivity_analysis_runtime()
