import random
import numpy as np
import time
from deap import base, creator, tools, algorithms
from InsectSim import run_simulation

# Define fitness function
def evaluate_agent(individual):
    scout_percentage, aggressiveness, max_hearing_distance, agent_speed = individual

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
            base_resource_speed=0.1,
            predator_speed=0.25,
            steps=500,
            detection_radius=5,
            resource_quantity=300,
            aggressiveness=aggressiveness,
            num_predators=2,
            create_csv=False
        )
        fitness_values.append(fitness)

    average_fitness = np.mean(fitness_values)
    return average_fitness,

# Set up genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_scout_percentage", random.uniform, 0.0, 1.0)
toolbox.register("attr_aggressiveness", random.uniform, 0.0, 1.0)
toolbox.register("attr_max_hearing_distance", random.uniform, 1.0, 60.0)
toolbox.register("attr_agent_speed", random.uniform, 0.2, 2.0)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_scout_percentage, toolbox.attr_aggressiveness,
                  toolbox.attr_max_hearing_distance, toolbox.attr_agent_speed), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=20)

toolbox.register("evaluate", evaluate_agent)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate_and_clamp(individual, mu, sigma, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            # Clamp the value to the specified bounds
            if i == 0:  # scout_percentage
                individual[i] = min(max(individual[i], 0.0), 1.0)
            elif i == 1:  # aggressiveness
                individual[i] = min(max(individual[i], 0.0), 1.0)
            elif i == 2:  # max_hearing_distance
                individual[i] = min(max(individual[i], 1.0), 60.0)
            elif i == 3:  # agent_speed
                individual[i] = min(max(individual[i], 0.2), 2.0)
    return individual,

toolbox.register("mutate", mutate_and_clamp, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the genetic algorithm
def run_genetic_algorithm():
    population = toolbox.population()
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    print(f"Starting Genetic Algorithm with {ngen} generations and population size of {len(population)}")

    start_time = time.time()

    for gen in range(1, ngen + 1):
        print(f"Generation {gen}/{ngen}")
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, 1, verbose=False)
        best_ind = tools.selBest(population, k=1)[0]
        print(f"Best individual at generation {gen}: {best_ind}, Fitness: {best_ind.fitness.values[0]}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Genetic Algorithm completed in {elapsed_time:.2f} seconds")

    best_ind = tools.selBest(population, k=1)[0]
    print("Best individual is:", best_ind)
    print("Best fitness:", best_ind.fitness.values[0])
    return best_ind

if __name__ == "__main__":
    best_params = run_genetic_algorithm()
