from InsectSim import run_simulation
from SmoothVideo import visualize_simulation

def main():

    # Fixed parameters --> specific to world
    grid_size = 80
    num_agents = 100
    resource_positions = [(10, 10), (70, 60)]
    base_positions = [(50, 50)]
    predator_radius = 5
    hazard_positions = [(25, 25), (60, 60)]
    hazard_radius = 7
    base_resource_speed = 0.1
    predator_speed = 0.25
    detection_radius = 5
    resource_quantity = 300  # Initial quantity of each resource
    num_predators = 2  # Number of predators

    # Parameters of agents --> tunable
    scout_percentage = 0.1
    aggressiveness = 0.5  # Aggressiveness level of the agents
    max_hearing_distance = 10
    agent_speed = 1.0

    # Specific to simulation
    steps = 500

    fitness, filename = run_simulation(grid_size, num_agents, scout_percentage, resource_positions,
                                       base_positions, max_hearing_distance, predator_radius,
                                       hazard_positions, hazard_radius, agent_speed,
                                       base_resource_speed, predator_speed, steps, detection_radius, resource_quantity, aggressiveness, num_predators)

    print(f'The value for Fitness is: {fitness}')
    visualize_simulation(filename, detection_radius, hazard_radius)

if __name__ == "__main__":
    main()