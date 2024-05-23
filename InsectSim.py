import random
import numpy as np
import pandas as pd
import cv2
import os


class Agent:
    def __init__(self, grid_size, max_hearing_distance):
        self.x = random.uniform(0, grid_size - 1)
        self.y = random.uniform(0, grid_size - 1)
        self.grid_size = grid_size
        self.max_hearing_distance = max_hearing_distance
        self.counters = {'resource': random.randint(0, grid_size - 1), 'base': random.randint(0, grid_size - 1)}
        if random.randint(0,1) == 1:
            self.current_target = 'resource'
        else:
            self.current_target = 'base'
        self.direction = random.uniform(0, 2 * np.pi)  # Angle in radians

    def move(self):
        dx = np.cos(self.direction)
        dy = np.sin(self.direction)
        new_x = self.x + dx
        new_y = self.y + dy

        # Bounce off the boundaries
        if new_x < 0 or new_x >= self.grid_size:
            self.direction = np.pi - self.direction
            new_x = self.x  # Prevent getting stuck by not updating position in the current step
        else:
            self.x = new_x

        if new_y < 0 or new_y >= self.grid_size:
            self.direction = -self.direction
            new_y = self.y  # Prevent getting stuck by not updating position in the current step
        else:
            self.y = new_y

    def step(self):
        self.move()
        for key in self.counters:
            self.counters[key] += 1

    def check_position(self, resource_positions, base_position):
        if (int(self.x), int(self.y)) in resource_positions:
            self.counters['resource'] = 0
            self.current_target = 'base'
            self.direction = (self.direction + np.pi) % (2 * np.pi)  # Invert direction
        if (int(self.x), int(self.y)) == base_position:
            self.counters['base'] = 0
            self.current_target = 'resource'
            self.direction = (self.direction + np.pi) % (2 * np.pi)  # Invert direction

    def shout(self):
        shout_value_resource = self.counters['resource'] + self.max_hearing_distance
        shout_value_base = self.counters['base'] + self.max_hearing_distance
        return self.x, self.y, shout_value_resource, shout_value_base

    def listen(self, shouts):
        for shout in shouts:
            sx, sy, shout_value_resource, shout_value_base = shout
            distance = abs(self.x - sx) + abs(self.y - sy)

            if distance <= self.max_hearing_distance:
                if shout_value_resource < self.counters['resource']:
                    self.counters['resource'] = shout_value_resource
                    if self.current_target == 'resource':
                        self.set_direction_towards(sx, sy)

                if shout_value_base < self.counters['base']:
                    self.counters['base'] = shout_value_base
                    if self.current_target == 'base':
                        self.set_direction_towards(sx, sy)

    def set_direction_towards(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        self.direction = np.arctan2(dy, dx)


def run_simulation(grid_size, num_agents, resource_positions, base_position, max_hearing_distance, steps):
    agents = [Agent(grid_size, max_hearing_distance) for _ in range(num_agents)]
    data = []

    for step in range(steps):
        shouts = []
        for agent in agents:
            agent.step()
            agent.check_position(resource_positions, base_position)
            shouts.append(agent.shout())

        for agent in agents:
            agent.listen(shouts)

        step_data = {
            'step': step + 1,
            'agents': [(int(agent.x), int(agent.y)) for agent in agents],
            'resources': resource_positions,
            'base': base_position
        }
        data.append(step_data)

        print(f"Step {step + 1}:")
        # for idx, agent in enumerate(agents):
        #     print(f"  Agent {idx}: Position ({int(agent.x)}, {int(agent.y)}), Counters: {agent.counters}")

    filename = save_simulation_data(data, grid_size, num_agents, max_hearing_distance, steps)
    return filename


def save_simulation_data(data, grid_size, num_agents, max_hearing_distance, steps):
    records = []
    for entry in data:
        step = entry['step']
        for idx, pos in enumerate(entry['agents']):
            records.append({
                'step': step,
                'agent_id': idx,
                'x': pos[0],
                'y': pos[1],
                'type': 'agent'
            })
        for pos in entry['resources']:
            records.append({
                'step': step,
                'agent_id': None,
                'x': pos[0],
                'y': pos[1],
                'type': 'resource'
            })
        records.append({
            'step': step,
            'agent_id': None,
            'x': entry['base'][0],
            'y': entry['base'][1],
            'type': 'base'
        })

    df = pd.DataFrame(records)
    filename = f'simulation_grid{grid_size}_agents{num_agents}_hearing{max_hearing_distance}_steps{steps}.csv'
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename

def visualize_simulation(filename):
    # Load the CSV data
    df = pd.read_csv(filename)

    # Extract the grid size and number of steps from the filename
    parts = filename.split('_')
    grid_size = int(parts[1].replace('grid', ''))
    num_agents = int(parts[2].replace('agents', ''))
    max_hearing_distance = int(parts[3].replace('hearing', ''))
    steps = int(parts[4].replace('steps', '').replace('.csv', ''))

    # Set up the video writer
    scale = 50  # scale to increase the size of the visualization
    grid_visual_size = grid_size * scale
    frame_rate = 60  # Increased frame rate for smoother animation
    video_writer = cv2.VideoWriter('simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                   (grid_visual_size, grid_visual_size))

    for step in range(1, steps + 1):
        frame = np.ones((grid_visual_size, grid_visual_size, 3), dtype=np.uint8) * 255

        step_data = df[df['step'] == step]

        # Draw the base
        base = step_data[step_data['type'] == 'base'].iloc[0]
        base_x, base_y = base['x'], base['y']
        cv2.circle(frame, (base_x * scale + scale // 2, base_y * scale + scale // 2), scale // 3, (0, 0, 255), -1)

        # Draw the resources
        resources = step_data[step_data['type'] == 'resource']
        for _, resource in resources.iterrows():
            res_x, res_y = resource['x'], resource['y']
            cv2.circle(frame, (res_x * scale + scale // 2, res_y * scale + scale // 2), scale // 3, (0, 255, 0), -1)

        # Draw the agents
        agents = step_data[step_data['type'] == 'agent']
        for _, agent in agents.iterrows():
            agent_x, agent_y = agent['x'], agent['y']
            cv2.circle(frame, (int(agent_x * scale + scale // 2), int(agent_y * scale + scale // 2)), scale // 4, (255, 0, 0), -1)

        # Write the frame to the video
        video_writer.write(frame)

        # Display the frame
        cv2.imshow('Simulation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Reduced delay for higher frame rate
            break

    # Release the video writer and close OpenCV windows
    video_writer.release()
    cv2.destroyAllWindows()

def main():
    # Configuration
    grid_size = 100
    num_agents = 200
    resource_positions = [(50, 10), [90, 80]]
    base_position = (60, 55)
    max_hearing_distance = 25
    steps = 500

    # Run simulation
    filename = run_simulation(grid_size, num_agents, resource_positions, base_position, max_hearing_distance, steps)

    # Visualize simulation
    visualize_simulation(filename)

if __name__ == "__main__":
    main()
