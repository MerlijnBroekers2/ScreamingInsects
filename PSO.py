import numpy as np
from pyswarm import pso
from InsectSim import run_simulation
from SmoothVideo import visualize_simulation

def objective_function(params):
    scout_percentage, aggressiveness, max_hearing_distance = params
    scout_percentage = np.clip(scout_percentage, 0, 1)
    aggressiveness = np.clip(aggressiveness, 0, 1)
    max_hearing_distance = np.clip(max_hearing_distance, 0, 100)

    grid_size = 100
    steps = 1000
    num_agents = 200
    agent_speed = 1.0
    detection_radius = 6
    resource_positions = [(60, 80), (90, 20)]
    resource_quantity = 250
    resource_speed = 0.05
    base_positions = [(10, 25)]
    base_speed = 0.0
    num_predators = 5
    predator_speed = 0.46
    predator_radius = 3
    hazard_positions = [[40, 40], [55, 60]]
    hazard_radius = 5
    safe_zone = (0, 0, 20, 50)
    penalty = -10

    fitness, _ = run_simulation(grid_size, num_agents, scout_percentage, resource_positions,
                                base_positions, max_hearing_distance, predator_radius,
                                hazard_positions, hazard_radius, agent_speed,
                                base_speed, resource_speed, predator_speed, steps, detection_radius,
                                resource_quantity, aggressiveness, num_predators, safe_zone, penalty,
                                create_csv=False)
    return -fitness  # We negate the fitness because PSO minimizes the objective function

def main():
    # PSO bounds for the parameters
    lb = [0.0, 0.0, 0.0]  # Lower bounds: scout_percentage, aggressiveness, max_hearing_distance
    ub = [0.2, 1.0, 100.0]  # Upper bounds: scout_percentage, aggressiveness, max_hearing_distance

    # Perform PSO to find the best parameters
    best_params, best_fitness = pso(objective_function, lb, ub, swarmsize=30, maxiter=100, c1=1.5, c2=1.5, w=0.7)

    scout_percentage, aggressiveness, max_hearing_distance = best_params
    scout_percentage = np.clip(scout_percentage, 0, 1)
    aggressiveness = np.clip(aggressiveness, 0, 1)
    max_hearing_distance = np.clip(max_hearing_distance, 0, 100)

    print(f'Optimized parameters: scout_percentage={scout_percentage}, aggressiveness={aggressiveness}, max_hearing_distance={max_hearing_distance}')
    print(f'Best fitness value: {-best_fitness}')

    # Run the battlefield simulation with optimized parameters
    grid_size = 100
    steps = 1000
    num_agents = 200
    agent_speed = 1.0
    detection_radius = 6
    resource_positions = [(60, 80), (90, 20)]
    resource_quantity = 250
    resource_speed = 0.05
    base_positions = [(10, 25)]
    base_speed = 0.0
    num_predators = 5
    predator_speed = 0.46
    predator_radius = 3
    hazard_positions = [[40, 40], [55, 60]]
    hazard_radius = 5
    safe_zone = (0, 0, 20, 50)
    penalty = -10

    fitness, filename = run_simulation(grid_size, num_agents, scout_percentage, resource_positions,
                                       base_positions, max_hearing_distance, predator_radius,
                                       hazard_positions, hazard_radius, agent_speed,
                                       base_speed, resource_speed, predator_speed, steps, detection_radius,
                                       resource_quantity, aggressiveness, num_predators, safe_zone, penalty,
                                       create_csv=True)

    # Output the fitness value and visualize the simulation
    print(f'The value for Fitness is: {fitness}')
    visualize_simulation(filename, detection_radius, hazard_radius, safe_zone)

if __name__ == "__main__":
    main()
