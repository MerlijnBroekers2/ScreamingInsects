from InsectSim import run_simulation
from SmoothVideo import visualize_simulation

def main():

    # Battlefield grid parameters
    grid_size = 100
    steps = 500

    # Drone agent parameters
    num_agents = 100
    scout_percentage = 0.1 # Percentage of scout drones
    aggressiveness = 0.5  # Aggressiveness level of the drones
    max_hearing_distance = 27  # Communication range of the drones
    agent_speed = 1.0  # Speed of the drones
    detection_radius = 3  # Detection radius of the drones

    # Friendly troop positions (resources) parameters
    resource_positions = [(90, 50)]  # Locations of friendly troops
    resource_quantity = 300  # Initial supply quantity needed by each troop
    resource_speed = 0.00  # Movement speed of friendly troops (if applicable)

    # Base (supply depot) parameters
    base_positions = [(10, 50)]  # Locations of supply depots
    base_speed = 0.0  # Movement speed of supply depots (if applicable)

    # Enemy combatant (predator) parameters
    num_predators = 0  # Number of enemy combatants
    predator_speed = 0.5  # Speed of enemy combatants
    predator_radius = 10  # Detection radius of enemy combatants

    # Minefield (hazard) parameters
    hazard_positions = []  # Locations of minefields
    hazard_radius = 15  # Detection radius of minefields

    # Safe zone parameters (e.g., secured supply routes)
    safe_zone = (0, 0, 0, 0)  # (x, y, width, height) of the safe zone

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
    visualize_simulation(filename, detection_radius, hazard_radius, safe_zone)

if __name__ == "__main__":
    main()
