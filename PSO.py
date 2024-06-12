import numpy as np
from pyswarm import pso
from InsectSim import run_simulation

# Define the objective function
def evaluate_agent(params):
    scout_percentage, aggressiveness, max_hearing_distance, agent_speed = params

    # Run the simulation multiple times to average the fitness
    num_runs = 5  # Number of runs to average
    fitness_values = []
    for _ in range(num_runs):
        fitness, _ = run_simulation(
            grid_size=80,
            num_agents=100,
            scout_percentage=scout_percentage,
            resource_positions=[(10, 10), (70, 60)],
            base_positions=[(50, 50)],
            max_hearing_distance=max_hearing_distance,
            predator_radius=5,
            hazard_positions=[(25, 25), (60, 60)],
            hazard_radius=7,
            agent_speed=agent_speed,
            base_speed=0.1,
            resource_speed=0.1,
            predator_speed=0.25,
            steps=500,
            detection_radius=5,
            resource_quantity=300,
            aggressiveness=aggressiveness,
            num_predators=2,
            safe_zone=None,
            penalty=-10,
            create_csv=False
        )
        fitness_values.append(fitness)

    average_fitness = np.mean(fitness_values)
    return -average_fitness  # Negative because PSO minimizes the function

# Define the parameter bounds
lb = [0.0, 0.0, 1.0, 0.2]  # Lower bounds for scout_percentage, aggressiveness, max_hearing_distance, agent_speed
ub = [1.0, 1.0, 60.0, 2.0]  # Upper bounds for scout_percentage, aggressiveness, max_hearing_distance, agent_speed

# Run PSO
best_params, best_fitness = pso(evaluate_agent, lb, ub, swarmsize=5, maxiter=5, debug=True)

print("Best parameters found: ", best_params)
print("Best fitness value: ", -best_fitness)  # Convert back to positive
