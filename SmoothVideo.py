import random
import numpy as np
import pandas as pd
import cv2
import os

def save_simulation_frames(filename, output_folder='simulation_frames'):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the CSV data
    df = pd.read_csv(filename)

    # Extract the grid size and number of steps from the filename
    parts = filename.split('_')
    grid_size = int(parts[1].replace('grid', ''))
    steps = int(parts[4].replace('steps', '').replace('.csv', ''))

    scale = 50  # scale to increase the size of the visualization
    grid_visual_size = grid_size * scale

    for step in range(1, steps + 1):
        frame = np.ones((grid_visual_size, grid_visual_size, 3), dtype=np.uint8) * 255

        step_data = df[df['step'] == step]

        # Draw the base
        base = step_data[step_data['type'] == 'base'].iloc[0]
        base_x, base_y = base['x'], base['y']
        cv2.circle(frame, (int(base_x * scale + scale // 2), int(base_y * scale + scale // 2)), scale // 3, (0, 0, 255), -1)

        # Draw the resources
        resources = step_data[step_data['type'] == 'resource']
        for _, resource in resources.iterrows():
            res_x, res_y = resource['x'], resource['y']
            cv2.circle(frame, (int(res_x * scale + scale // 2), int(res_y * scale + scale // 2)), scale // 3, (0, 255, 0), -1)

        # Draw the agents
        agents = step_data[step_data['type'] == 'agent']
        for _, agent in agents.iterrows():
            agent_x, agent_y = agent['x'], agent['y']
            cv2.circle(frame, (int(agent_x * scale + scale // 2), int(agent_y * scale + scale // 2)), scale // 4, (255, 0, 0), -1)

        # Save the frame as an image
        cv2.imwrite(os.path.join(output_folder, f'frame_{step:04d}.png'), frame)

    print(f"Frames saved in folder: {output_folder}")

def create_video_from_frames(output_folder='simulation_frames', output_video='simulation.mp4', frame_rate=30):
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.png')])

    # Check if there are any frames to process
    if not frame_files:
        print("No frames found in the folder.")
        return

    # Read the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(output_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Set up the video writer
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(output_folder, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created: {output_video}")

create_video_from_frames()