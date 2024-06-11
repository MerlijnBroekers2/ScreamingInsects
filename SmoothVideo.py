import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

def visualize_simulation(filename, detection_radius, hazard_radius, safe_zone=None):
    df = pd.read_csv(filename)
    parts = filename.split('_')
    grid_size = int(parts[1].replace('grid', ''))
    num_agents = int(parts[2].replace('agents', ''))
    num_scouts = int(parts[3].replace('scouts', ''))
    num_resources = int(parts[4].replace('resources', ''))
    num_bases = int(parts[5].replace('bases', ''))
    max_hearing_distance = int(parts[6].replace('hearing', ''))
    steps = int(parts[7].replace('steps', ''))
    num_predators = int(parts[8].replace('predators', '').replace('.csv', ''))

    scale = 50
    grid_visual_size = grid_size * scale
    frame_rate = 30
    video_writer = cv2.VideoWriter('simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                   (grid_visual_size, grid_visual_size))

    for step in range(1, steps + 1):
        frame = np.ones((grid_visual_size, grid_visual_size, 3), dtype=np.uint8) * 255

        # Draw safe zone if defined
        if safe_zone:
            safe_zone_x, safe_zone_y, safe_zone_width, safe_zone_height = safe_zone
            top_left = (safe_zone_x * scale, safe_zone_y * scale)
            bottom_right = ((safe_zone_x + safe_zone_width) * scale, (safe_zone_y + safe_zone_height) * scale)
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
            alpha = 0.3  # Transparency factor
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        step_data = df[df['step'] == step]

        # Draw hazards first
        hazards = step_data[step_data['type'] == 'hazard']
        for _, hazard in hazards.iterrows():
            hazard_x, hazard_y = hazard['x'], hazard['y']
            cv2.circle(frame, (hazard_x * scale + scale // 2, hazard_y * scale + scale // 2), (hazard_radius - 1) * scale, (128, 128, 128), -1)

        # Draw predators
        for idx in range(num_predators):
            predator = step_data[step_data['type'] == f'predator_{idx}'].iloc[0]
            predator_x, predator_y = predator['x'], predator['y']
            cv2.circle(frame, (predator_x * scale + scale // 2, predator_y * scale + scale // 2), scale // 2, (0, 0, 0), -1)

        # Draw agents
        agents = step_data[step_data['type'] == 'agent']
        for _, agent in agents.iterrows():
            agent_x, agent_y = agent['x'], agent['y']
            cv2.circle(frame, (int(agent_x * scale + scale // 2), int(agent_y * scale + scale // 2)), scale // 4, (255, 0, 0), -1)

        scouts = step_data[step_data['type'] == 'scout']
        for _, scout in scouts.iterrows():
            scout_x, scout_y = scout['x'], scout['y']
            cv2.circle(frame, (int(scout_x * scale + scale // 2), int(scout_y * scale + scale // 2)), scale // 4, (0, 0, 255), -1)

        # Draw bases/resources last
        bases = step_data[step_data['type'] == 'base']
        for _, base in bases.iterrows():
            base_x, base_y = base['x'], base['y']
            cv2.circle(frame, (base_x * scale + scale // 2, base_y * scale + scale // 2), (detection_radius - 1) * scale, (0, 0, 255), -1)

        resources = step_data[step_data['type'] == 'resource']
        for _, resource in resources.iterrows():
            res_x, res_y = resource['x'], resource['y']
            cv2.circle(frame, (res_x * scale + scale // 2, res_y * scale + scale // 2), (detection_radius - 1) * scale, (0, 255, 0), -1)

        video_writer.write(frame)

        cv2.imshow('Simulation', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    video_writer.release()
    cv2.destroyAllWindows()
