import random
import numpy as np
import pandas as pd


class Agent:
    def __init__(self, grid_size, max_hearing_distance, speed, detection_radius=2, aggressiveness=0.5,
                 hazard_positions=[], hazard_radius=0):
        while True:
            self.x = random.uniform(2, grid_size - 2)
            self.y = random.uniform(2, grid_size - 2)
            if all(np.sqrt((self.x - hx) ** 2 + (self.y - hy) ** 2) > hazard_radius for hx, hy in hazard_positions):
                break
        self.grid_size = grid_size
        self.max_hearing_distance = max_hearing_distance
        self.speed = speed
        self.detection_radius = detection_radius
        self.aggressiveness = aggressiveness
        self.counters = {'resource': random.randint(0, grid_size - 1), 'base': random.randint(0, grid_size - 1)}
        if random.randint(0, 1) == 1:
            self.current_target = 'resource'
        else:
            self.current_target = 'base'
        self.direction = random.uniform(0, 2 * np.pi)
        self.collected_resources = 0
        self.carrying_resource = False  # Track if the agent is carrying a resource
        self.alive = True

    def move(self):
        dx = self.speed * np.cos(self.direction)
        dy = self.speed * np.sin(self.direction)
        new_x = self.x + dx
        new_y = self.y + dy

        if new_x < 0 or new_x >= self.grid_size:
            self.direction = np.pi - self.direction
            new_x = self.x
        else:
            self.x = new_x

        if new_y < 0 or new_y >= self.grid_size:
            self.direction = -self.direction
            new_y = self.y
        else:
            self.y = new_y

    def step(self):
        self.move()
        for key in self.counters:
            self.counters[key] += 1

    def check_position(self, resources, base_position):
        if self.carrying_resource:
            if np.sqrt((self.x - base_position[0]) ** 2 + (self.y - base_position[1]) ** 2) <= self.detection_radius:
                self.counters['base'] = 0
                self.current_target = 'resource'
                self.direction = (self.direction + np.pi) % (2 * np.pi)
                self.carrying_resource = False
                self.collected_resources += 1  # Award point for delivering the resource
                return

        for resource in resources:
            if np.sqrt((self.x - resource.x) ** 2 + (self.y - resource.y) ** 2) <= self.detection_radius:
                self.counters['resource'] = 0
                self.current_target = 'base'
                self.direction = (self.direction + np.pi) % (2 * np.pi)
                self.carrying_resource = True  # Pick up the resource
                resource.quantity -= 1  # Reduce resource quantity by one
                if resource.quantity <= 0:
                    resources.remove(resource)  # Remove depleted resource
                return

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

    def set_direction_away(self, px, py):
        dx = self.x - px
        dy = self.y - py
        self.direction = np.arctan2(dy, dx)

    def react_to_predator(self, predators, predator_radius):
        for predator in predators:
            distance = np.sqrt((self.x - predator.x) ** 2 + (self.y - predator.y) ** 2)
            if distance <= predator_radius:
                if random.random() > self.aggressiveness:  # Only avoid predator based on aggressiveness level
                    self.set_direction_away(predator.x, predator.y)
                if distance <= predator_radius / 2:  # Agent dies if too close to the predator
                    self.alive = False

    def react_to_hazard(self, hazard_positions, hazard_radius):
        for hazard_x, hazard_y in hazard_positions:
            distance = np.sqrt((self.x - hazard_x) ** 2 + (self.y - hazard_y) ** 2)
            if distance <= hazard_radius:
                self.avoid_hazard(hazard_positions, hazard_radius)
                return True
        return False

    def avoid_hazard(self, hazard_positions, hazard_radius):
        best_direction = self.direction
        best_distance = float('inf')
        steps_to_check = 8  # Number of directions to check for the best path

        for i in range(steps_to_check):
            angle = self.direction + (2 * np.pi * i / steps_to_check)
            dx = self.speed * np.cos(angle)
            dy = self.speed * np.sin(angle)
            new_x = self.x + dx
            new_y = self.y + dy

            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                continue  # Skip if out of bounds

            min_distance_to_hazard = min(np.sqrt((new_x - hx) ** 2 + (new_y - hy) ** 2) for hx, hy in hazard_positions)
            if min_distance_to_hazard > hazard_radius:
                if min_distance_to_hazard < best_distance:
                    best_distance = min_distance_to_hazard
                    best_direction = angle

        # Move in the best direction found
        self.direction = best_direction
        dx = self.speed * np.cos(self.direction)
        dy = self.speed * np.sin(self.direction)
        self.x += dx
        self.y += dy


class ScoutAgent(Agent):
    def __init__(self, grid_size, max_hearing_distance, speed, detection_radius=2, aggressiveness=0.5,
                 hazard_positions=[], hazard_radius=0):
        super().__init__(grid_size, max_hearing_distance, speed, detection_radius, aggressiveness, hazard_positions,
                         hazard_radius)

    def check_position(self, resources, base_position):
        for resource in resources:
            if np.sqrt((self.x - resource.x) ** 2 + (self.y - resource.y) ** 2) <= self.detection_radius:
                self.counters['resource'] = 0

    def shout(self):
        shout_value_resource = self.counters['resource'] + self.max_hearing_distance
        return self.x, self.y, shout_value_resource, 10000

    def listen(self, shouts):
        for shout in shouts:
            sx, sy, shout_value_resource, shout_value_base = shout
            distance = abs(self.x - sx) + abs(self.y - sy)

            if distance <= self.max_hearing_distance:
                if shout_value_resource < self.counters['resource']:
                    self.counters['resource'] = shout_value_resource

                if shout_value_base < self.counters['base']:
                    self.counters['base'] = shout_value_base


class Predator:
    def __init__(self, grid_size, speed, safe_zone=None):
        while True:
            self.x = random.uniform(10, grid_size - 10)
            self.y = random.uniform(10, grid_size - 10)
            if safe_zone:
                safe_zone_x, safe_zone_y, safe_zone_width, safe_zone_height = safe_zone
                if not (
                        safe_zone_x <= self.x <= safe_zone_x + safe_zone_width and safe_zone_y <= self.y <= safe_zone_y + safe_zone_height):
                    break
            else:
                break

        self.grid_size = grid_size
        self.speed = speed
        self.direction = random.uniform(0, 2 * np.pi)
        self.safe_zone = safe_zone

    def move(self):
        dx = self.speed * np.cos(self.direction)
        dy = self.speed * np.sin(self.direction)
        new_x = self.x + dx
        new_y = self.y + dy

        if new_x < 0 or new_x >= self.grid_size:
            self.direction = np.pi - self.direction
            new_x = self.x
        else:
            self.x = new_x

        if new_y < 0 or new_y >= self.grid_size:
            self.direction = -self.direction
            new_y = self.y
        else:
            self.y = new_y

        # Avoid safe zone
        if self.safe_zone:
            safe_zone_x, safe_zone_y, safe_zone_width, safe_zone_height = self.safe_zone
            if (safe_zone_x <= new_x <= safe_zone_x + safe_zone_width) and (
                    safe_zone_y <= new_y <= safe_zone_y + safe_zone_height):
                self.direction = (self.direction + np.pi) % (2 * np.pi)

    def step(self):
        self.move()


class Hazard:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius


class MovableObject:
    def __init__(self, grid_size, speed, x=None, y=None, hazard_positions=[], hazard_radius=0):
        self.grid_size = grid_size
        self.speed = speed

        while True:
            if x is not None and y is not None:
                self.x = x
                self.y = y
            else:
                self.x = random.uniform(0, grid_size - 1)
                self.y = random.uniform(0, grid_size - 1)

            if all(np.sqrt((self.x - hx) ** 2 + (self.y - hy) ** 2) > hazard_radius for hx, hy in hazard_positions):
                break

        self.direction = random.uniform(0, 2 * np.pi)  # Angle in radians

    def move(self, hazard_positions, hazard_radius):
        dx = self.speed * np.cos(self.direction)
        dy = self.speed * np.sin(self.direction)
        new_x = self.x + dx
        new_y = self.y + dy

        # Bounce off the boundaries
        if new_x < 0 or new_x >= self.grid_size or any(
                np.sqrt((new_x - hx) ** 2 + (new_y - hy) ** 2) <= hazard_radius for hx, hy in hazard_positions):
            self.direction = np.pi - self.direction
            new_x = self.x  # Prevent getting stuck by not updating position in the current step
        else:
            self.x = new_x

        if new_y < 0 or new_y >= self.grid_size or any(
                np.sqrt((new_x - hx) ** 2 + (new_y - hy) ** 2) <= hazard_radius for hx, hy in hazard_positions):
            self.direction = -self.direction
            new_y = self.y  # Prevent getting stuck by not updating position in the current step
        else:
            self.y = new_y

    def step(self, hazard_positions, hazard_radius):
        self.move(hazard_positions, hazard_radius)


class Resource(MovableObject):
    def __init__(self, grid_size, speed, quantity, x=None, y=None, hazard_positions=[], hazard_radius=0):
        super().__init__(grid_size, speed, x, y, hazard_positions, hazard_radius)
        self.quantity = quantity


def generate_new_resource_position(grid_size, hazard_positions, hazard_radius):
    while True:
        new_x = random.uniform(10, grid_size - 10)
        new_y = random.uniform(10, grid_size - 10)
        if all(np.sqrt((new_x - hx) ** 2 + (new_y - hy) ** 2) > hazard_radius for hx, hy in hazard_positions):
            return new_x, new_y


def run_simulation(grid_size, num_agents, scout_percentage, resource_positions, base_positions, max_hearing_distance,
                   predator_radius,
                   hazard_positions, hazard_radius, agent_speed, base_speed, resource_speed, predator_speed, steps,
                   detection_radius=2, resource_quantity=10, aggressiveness=0.5, num_predators=1, safe_zone=None,
                   penalty=-10, create_csv=True):
    num_scouts = int(num_agents * scout_percentage)
    num_agents = int(num_agents - num_scouts)
    agents = [Agent(grid_size, max_hearing_distance, agent_speed, detection_radius, aggressiveness, hazard_positions,
                    hazard_radius) for _ in range(num_agents)]
    scouts = [
        ScoutAgent(grid_size, max_hearing_distance, agent_speed, detection_radius, aggressiveness, hazard_positions,
                   hazard_radius) for _ in range(num_scouts)]
    resources = [
        Resource(grid_size, resource_speed, resource_quantity, x=pos[0], y=pos[1], hazard_positions=hazard_positions,
                 hazard_radius=hazard_radius) for pos in resource_positions]
    bases = [MovableObject(grid_size, base_speed, x=pos[0], y=pos[1], hazard_positions=hazard_positions,
                           hazard_radius=hazard_radius) for pos in base_positions]
    predators = [Predator(grid_size, predator_speed, safe_zone=safe_zone) for _ in range(num_predators)]
    hazards = [Hazard(pos[0], pos[1], hazard_radius) for pos in hazard_positions]
    data = []

    for step in range(steps):
        for predator in predators:
            predator.step()
        for resource in resources:
            resource.step(hazard_positions, hazard_radius)
        for base in bases:
            base.step(hazard_positions, hazard_radius)

        shouts = []
        alive_agents = []
        for agent in agents + scouts:
            if agent.alive:
                agent.step()
                agent.check_position(resources, (int(bases[0].x), int(bases[0].y)))
                agent.react_to_predator(predators, predator_radius)
                if agent.alive:
                    shouts.append(agent.shout())
                    alive_agents.append(agent)
            for hazard in hazards:
                agent.react_to_hazard(hazard_positions, hazard_radius)

        for agent in alive_agents:
            agent.listen(shouts)
            agent.react_to_predator(predators, predator_radius)
            for hazard in hazards:
                agent.react_to_hazard(hazard_positions, hazard_radius)

        agents = [agent for agent in agents if agent.alive]
        scouts = [scout for scout in scouts if scout.alive]

        while len(resources) < len(resource_positions):
            new_x, new_y = generate_new_resource_position(grid_size, hazard_positions, hazard_radius)
            resources.append(Resource(grid_size, resource_speed, resource_quantity, x=new_x, y=new_y,
                                      hazard_positions=hazard_positions, hazard_radius=hazard_radius))

        step_data = {
            'step': step + 1,
            'agents': [(int(agent.x), int(agent.y)) for agent in agents],
            'scouts': [(int(agent.x), int(agent.y)) for agent in scouts],
            'resources': [(int(res.x), int(res.y)) for res in resources],
            'bases': [(int(base.x), int(base.y)) for base in bases],
            'predators': [(int(predator.x), int(predator.y)) for predator in predators],
            'hazards': [(hazard.x, hazard.y) for hazard in hazards]
        }
        data.append(step_data)

    collected_resources = sum(agent.collected_resources for agent in agents)
    num_agents_lost = num_agents + num_scouts - len(agents) - len(scouts)
    fitness = collected_resources + penalty * num_agents_lost

    if create_csv:
        filename = save_simulation_data(data, grid_size, num_agents, scout_percentage, len(resource_positions),
                                        len(base_positions), max_hearing_distance, steps, num_predators)
    else:
        filename = "no_save"

    print(f'collected_resources: {collected_resources}')
    print(f'num_agents_lost: {num_agents_lost}')
    print(f'fitness: {fitness}')

    return fitness, filename


def save_simulation_data(data, grid_size, num_agents, scout_percentage, num_resources, num_bases, max_hearing_distance,
                         steps, num_predators):
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
        for idx, pos in enumerate(entry['scouts']):
            records.append({
                'step': step,
                'agent_id': num_agents + idx,
                'x': pos[0],
                'y': pos[1],
                'type': 'scout'
            })
        for idx, pos in enumerate(entry['resources']):
            records.append({
                'step': step,
                'agent_id': num_agents + int(num_agents * scout_percentage) + idx,
                'x': pos[0],
                'y': pos[1],
                'type': 'resource'
            })
        for idx, pos in enumerate(entry['bases']):
            records.append({
                'step': step,
                'agent_id': num_agents + int(num_agents * scout_percentage) + num_resources + idx,
                'x': pos[0],
                'y': pos[1],
                'type': 'base'
            })
        for idx, pos in enumerate(entry['predators']):
            records.append({
                'step': step,
                'agent_id': None,
                'x': pos[0],
                'y': pos[1],
                'type': f'predator_{idx}'
            })
        for idx, pos in enumerate(entry['hazards']):
            records.append({
                'step': step,
                'agent_id': None,
                'x': pos[0],
                'y': pos[1],
                'type': 'hazard'
            })

    df = pd.DataFrame(records)
    filename = f'CSV/simulation_grid{grid_size}_agents{num_agents}_scouts{int(scout_percentage * 100)}_resources{num_resources}_bases{num_bases}_hearing{max_hearing_distance}_steps{steps}_predators{num_predators}.csv'
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename
