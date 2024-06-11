import numpy as np
import matplotlib.pyplot as plt
from InsectSim import run_simulation
import seaborn as sns


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
                hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                create_csv=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(fitness_matrix, aspect='auto', cmap='viridis', extent=[20, 400, 200, 40])
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
            hazard_radius=7, agent_speed=1.0, base_resource_speed=speed, predator_speed=0.25,
            steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            create_csv=False)
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
            hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=speed,
            steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            create_csv=False)
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
            hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=0.25,
            steps=500, detection_radius=radius, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            create_csv=False)
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
            hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=0.25,
            steps=steps, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
            create_csv=False)
        fitness_values.append(fitness)

    plt.figure()
    plt.plot(steps_values, fitness_values, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Average Fitness')
    plt.title('Fitness vs Steps of the Simulation')
    plt.grid(True)
    plt.show()


def average_fitness_over_runs(runs, **simulation_params):
    fitness_values = []
    for _ in range(runs):
        fitness, _ = run_simulation(**simulation_params)
        fitness_values.append(fitness)
    return np.mean(fitness_values)


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
                hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=0.25,
                steps=500, detection_radius=radius, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                create_csv=False)

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
                hazard_radius=7, agent_speed=speed, base_resource_speed=0.1, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                create_csv=False)

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
                hazard_radius=radius, agent_speed=1.0, base_resource_speed=0.1, predator_speed=speed,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                create_csv=False)

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
                hazard_radius=7, agent_speed=1.0, base_resource_speed=0.1, predator_speed=0.25,
                steps=500, detection_radius=5, resource_quantity=300, aggressiveness=0.5, num_predators=2,
                create_csv=False)

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
    base_resource_speed = 0.1
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
        agent_speed=agent_speed, base_resource_speed=base_resource_speed, predator_speed=predator_speed,
        steps=steps, detection_radius=detection_radius, resource_quantity=resource_quantity,
        aggressiveness=aggressiveness, num_predators=num_predators, create_csv=False)

    # Plot the distribution of fitness values
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fitness_values)
    plt.title('Impact of Randomization on Fitness')
    plt.ylabel('Fitness')
    plt.xlabel('Simulation Runs')
    plt.show()


if __name__ == "__main__":
    grid_size_vs_num_agents()
    # base_resource_speed_vs_fitness()
    # predator_speed_vs_fitness()
    # detection_radius_vs_fitness()
    # fitness_vs_steps()
    # grid_size_vs_detection_radius()
    # num_agents_vs_agent_speed()
    # predator_speed_vs_hazard_radius()
    # grid_size_vs_num_agents_3d()
    # analyze_randomization_impact()
