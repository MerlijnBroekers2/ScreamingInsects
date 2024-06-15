from InsectSim import run_simulation
from SmoothVideo import visualize_simulation

def main():

    # Battlefield grid parameters
    grid_size = 100  # 1 grid element = 1 km
    steps = 400

    # Drone agent parameters
    num_agents = 100
    scout_percentage = 0.1  # Percentage of scout drones
    aggressiveness = 0.0 # Aggressiveness level of the drones
    max_hearing_distance = 45  # Communication range of the drones in km (scaled to grid size)
    agent_speed = 1.0  # Speed of the drones (108 km/h, highest speed = 1.0)
    detection_radius = 6  # Detection radius of the drones in km

    # Friendly troop positions (resources) parameters
    resource_positions = [(60, 80), (90, 20)]  # Locations of friendly troops in grid coordinates
    resource_quantity = 250  # Initial supply quantity needed by each troop
    resource_speed = 0.05  # Movement speed of friendly troops (5 km/h scaled relative to 108 km/h)

    # Base (supply depot) parameters
    base_positions = [(10, 25)]  # Locations of supply depots in grid coordinates
    base_speed = 0.0  # Supply depots are stationary (0 km/h)

    # Enemy combatant (predator) parameters
    num_predators = 5  # Number of enemy combatants
    predator_speed = 0.46  # Speed of enemy combatants (50 km/h scaled relative to 108 km/h)
    predator_radius = 3  # Detection radius of enemy combatants in km

    # Minefield (hazard) parameters
    hazard_positions = [[40, 40], [55,60]]  # Locations of minefields in grid coordinates
    hazard_radius = 5 # Detection radius of minefields in km

    # Safe zone parameters (e.g., secured supply routes)
    safe_zone = (0, 0, 20, 50)  # (x, y, width, height) of the safe zone in grid coordinates

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