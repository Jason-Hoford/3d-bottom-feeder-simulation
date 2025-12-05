"""Main entry point for Bottom Feeder simulation."""
import pygame
import taichi as ti
import sys
import math

# Initialize Taichi before importing physics
ti.init(arch=ti.gpu)

from feeder_constants import (
    VIEW_WIDTH, VIEW_HEIGHT, FPS,
    CAMERA_MOVE_SPEED, CAMERA_ROTATION_SPEED, MOUSE_SENSITIVITY
)
from feeder_world import FeederWorld
from feeder_renderer import draw_feeder_world, draw_hud, Camera3D, draw_graphs

def main():
    pygame.init()
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption("Bottom Feeder Simulation (Taichi Accelerated)")
    clock = pygame.time.Clock()
    
    world = FeederWorld()
    camera = Camera3D()
    
    running = True
    pygame.mouse.set_visible(True) # Keep mouse visible
    pygame.event.set_grab(False)
    
    show_graph = True
    hunger_history = []
    consumption_history = []
    
    # Visualization mode
    from feeder_constants import VIS_MODE_NORMAL
    vis_mode = VIS_MODE_NORMAL
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        # Input Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    show_graph = not show_graph
                elif event.key == pygame.K_F5:
                    # Save State
                    from feeder_constants import SAVE_DIR, SAVE_PATTERN
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"feeder_save_{timestamp}.pkl"
                    world.save_state(filename)
                    print(f"Saved state to {filename}")
                elif event.key == pygame.K_F6:
                    # Load State (Load latest)
                    import glob
                    import os
                    saves = glob.glob("feeder_save_*.pkl")
                    if saves:
                        latest = max(saves, key=os.path.getctime)
                        print(f"Loading save: {latest}")
                        try:
                            world.load_state(latest)
                            print("Loaded successfully.")
                        except Exception as e:
                            print(f"Failed to load: {e}")
                    else:
                        print("No saves found.")
                elif event.key == pygame.K_l:
                    # Load latest TRAINED model
                    import glob
                    import os
                    import pickle
                    import random
                    from feeder_constants import WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH, INITIAL_FISH_COUNT
                    from feeder_entities import FeederFish
                    
                    models = glob.glob("trained_models/*.pkl")
                    if models:
                        latest_model = max(models, key=os.path.getctime)
                        print(f"Loading model: {latest_model}")
                        try:
                            with open(latest_model, "rb") as f:
                                loaded_net = pickle.load(f)
                                
                            # Repopulate
                            new_fish = []
                            for _ in range(INITIAL_FISH_COUNT):
                                net = loaded_net.clone()
                                net.mutate(0.02, 0.05) # Slight variation
                                fish = FeederFish(
                                    random.uniform(0, WORLD_WIDTH),
                                    random.uniform(0, WORLD_HEIGHT),
                                    random.uniform(0, WORLD_DEPTH),
                                    net
                                )
                                new_fish.append(fish)
                            world.fish = new_fish
                            print("Population replaced with trained agents.")
                        except Exception as e:
                            print(f"Failed to load model: {e}")
                    else:
                        print("No trained models found in 'trained_models/'")
                elif event.key == pygame.K_p:
                    # Cycle Visualization Mode
                    from feeder_constants import VIS_MODE_NORMAL, VIS_MODE_DENSITY, VIS_MODE_FLOW
                    vis_mode = (vis_mode + 1) % 3
                    modes = ["NORMAL", "DENSITY", "FLOW"]
                    print(f"Switched to {modes[vis_mode]} mode")

            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]: # Only rotate if left click held
                    dx, dy = event.rel
                    camera.angle_x += dx * MOUSE_SENSITIVITY
                    camera.angle_y += dy * MOUSE_SENSITIVITY
                    camera.angle_y = max(-1.5, min(1.5, camera.angle_y))
                
        # Camera Movement
        keys = pygame.key.get_pressed()
        
        # Calculate forward/right vectors for camera movement
        # Flat movement (xz plane)
        cx = math.sin(camera.angle_x)
        cz = math.cos(camera.angle_x)
        
        forward = (cx, 0, cz)
        right = (cz, 0, -cx)
        
        move_speed = CAMERA_MOVE_SPEED * dt
        if keys[pygame.K_LSHIFT]: move_speed *= 2.0
        
        if keys[pygame.K_w]:
            camera.center = (camera.center[0] - forward[0]*move_speed, camera.center[1], camera.center[2] - forward[2]*move_speed)
        if keys[pygame.K_s]:
            camera.center = (camera.center[0] + forward[0]*move_speed, camera.center[1], camera.center[2] + forward[2]*move_speed)
        if keys[pygame.K_a]:
            camera.center = (camera.center[0] - right[0]*move_speed, camera.center[1], camera.center[2] - right[2]*move_speed)
        if keys[pygame.K_d]:
            camera.center = (camera.center[0] + right[0]*move_speed, camera.center[1], camera.center[2] + right[2]*move_speed)
        if keys[pygame.K_q]:
            camera.center = (camera.center[0], camera.center[1] - move_speed, camera.center[2])
        if keys[pygame.K_e]:
            camera.center = (camera.center[0], camera.center[1] + move_speed, camera.center[2])
            
        # Zoom
        if keys[pygame.K_r]: camera.distance = max(100, camera.distance - move_speed * 2)
        if keys[pygame.K_f]: camera.distance += move_speed * 2
        
        # Update Simulation
        world.update(dt)
        
        # Update Density Grid if needed
        from feeder_constants import VIS_MODE_DENSITY
        if vis_mode == VIS_MODE_DENSITY:
            world.physics.update_density_grid()
        
        # Collect Stats
        if world.sim_time > 0:
            # Avg Hunger
            avg_hunger = 0.0
            if world.fish:
                avg_hunger = sum(f.hunger for f in world.fish) / len(world.fish)
            hunger_history.append((world.sim_time, avg_hunger))
            
            # Consumption (Rolling average over 1 second)
            # We track total eaten this frame
            frame_eaten = sum(world.physics.fish_eaten_count.to_numpy())
            
            # We need to store frame_eaten and sum up over last N frames (approx 1 sec)
            # Or just use a simple exponential moving average for display
            # Let's use EMA for the graph point
            
            # If we want "Food/Sec", we can just take frame_eaten / dt
            # But that is noisy.
            instant_rate = frame_eaten / dt if dt > 0 else 0
            
            # Smooth it
            if not hasattr(main, "smooth_rate"): main.smooth_rate = 0.0
            main.smooth_rate = 0.95 * main.smooth_rate + 0.05 * instant_rate
            
            consumption_history.append((world.sim_time, main.smooth_rate))
        
        # Render
        draw_feeder_world(screen, world, camera, vis_mode)
        draw_hud(screen, world, clock.get_fps())
        draw_graphs(screen, hunger_history, consumption_history, world.sim_time, show_graph)
        
        pygame.display.flip()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
