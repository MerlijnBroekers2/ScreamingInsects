import numpy as np
from pyswarm import pso
from InsectSim import run_simulation
from SmoothVideo import visualize_simulation

# Define the objective function
def evaluate_agent(params):
    scout_percentage, aggressiveness, max_hearing_distance = params

    # Case study parameters
    grid_size = 100
    num_agents = 50
    resource_positions = [(60, 80), (90, 20)]
    base_positions = [(10, 20)]
    predator_radius = 4
    hazard_positions = [(35, 35), (70, 60)]
    hazard_radius = 1
    agent_speed = 1.0  # Fixed agent speed
    base_speed = 0.0
    resource_speed = 0.05
    predator_speed = 0.05
    steps = 2000
    detection_radius = 5
    resource_quantity = 500
    num_predators = 5
    safe_zone = (0, 0, 20, 100)
    penalty = -10

    # Run the simulation multiple times to average the fitness
    num_runs = 20  # Number of runs to average
    fitness_values = []
    for _ in range(num_runs):
        fitness, _ = run_simulation(
            grid_size=grid_size,
            num_agents=num_agents,
            scout_percentage=scout_percentage,
            resource_positions=resource_positions,
            base_positions=base_positions,
            max_hearing_distance=max_hearing_distance,
            predator_radius=predator_radius,
            hazard_positions=hazard_positions,
            hazard_radius=hazard_radius,
            agent_speed=agent_speed,
            base_speed=base_speed,
            resource_speed=resource_speed,
            predator_speed=predator_speed,
            steps=steps,
            detection_radius=detection_radius,
            resource_quantity=resource_quantity,
            aggressiveness=aggressiveness,
            num_predators=num_predators,
            safe_zone=safe_zone,
            penalty=penalty,
            create_csv=False
        )
        fitness_values.append(fitness)

    average_fitness = np.mean(fitness_values)
    return -average_fitness  # Negative because PSO minimizes the function

# Define the parameter bounds
lb = [0.0, 0.0, 1.0]  # Lower bounds for scout_percentage, aggressiveness, max_hearing_distance
ub = [1.0, 1.0, 60.0]  # Upper bounds for scout_percentage, aggressiveness, max_hearing_distance

# Run PSO
best_params, best_fitness = pso(evaluate_agent, lb, ub, swarmsize=10, maxiter=10, debug=True)

print("Best parameters found: ", best_params)
print("Best fitness value: ", -best_fitness)  # Convert back to positive

# Run the main simulation with the best parameters found by PSO
def main():
    # Battlefield grid parameters
    grid_size = 100
    steps = 2000

    # Drone agent parameters
    num_agents = 50
    scout_percentage = best_params[0]  # Optimized scout percentage
    aggressiveness = best_params[1]  # Optimized aggressiveness level
    max_hearing_distance = best_params[2]  # Optimized communication range
    agent_speed = 1.0  # Fixed agent speed
    detection_radius = 5  # Detection radius of the drones

    # Friendly troop positions (resources) parameters
    resource_positions = [(60, 80), (90, 20)]  # Locations of friendly troops
    resource_quantity = 500  # Initial supply quantity needed by each troop
    resource_speed = 0.05  # Movement speed of friendly troops (if applicable)

    # Base (supply depot) parameters
    base_positions = [(10, 20)]  # Locations of supply depots
    base_speed = 0.0  # Movement speed of supply depots (if applicable)

    # Enemy combatant (predator) parameters
    num_predators = 5  # Number of enemy combatants
    predator_speed = 0.05  # Speed of enemy combatants
    predator_radius = 4  # Detection radius of enemy combatants

    # Minefield (hazard) parameters
    hazard_positions = [(35, 35), (70, 60)]  # Locations of minefields
    hazard_radius = 1  # Detection radius of minefields

    # Safe zone parameters (e.g., secured supply routes)
    safe_zone = (0, 0, 20, 100)  # (x, y, width, height) of the safe zone

    # Penalty for drones being shot down
    penalty = -10

    # Run the battlefield simulation
    fitness, filename = run_simulation(grid_size, num_agents, scout_percentage, resource_positions,
                                       base_positions, max_hearing_distance, predator_radius,
                                       hazard_positions, hazard_radius, agent_speed,
                                       base_speed, resource_speed, predator_speed, steps, detection_radius,
                                       resource_quantity, aggressiveness, num_predators, safe_zone, penalty)

    # Output the fitness value and visualize the simulation
    print(f'The value for Fitness is: {fitness}')


if __name__ == "__main__":
    main()
