"""Offline training script for Bottom Feeder simulation."""
import taichi as ti
import pickle
import time
from pathlib import Path

# Initialize Taichi (GPU)
ti.init(arch=ti.gpu)

from feeder_world import FeederWorld
from feeder_constants import INITIAL_FISH_COUNT
from feeder_logger import TrainingLogger

# Training Parameters
EPOCHS = 50
EPOCH_DURATION = 30.0  # Seconds of simulation time per epoch
DT = 1.0 / 60.0  # Simulation step size
MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)

def train():
    print(f"Starting offline training for {EPOCHS} epochs...")
    logger = TrainingLogger()
    
    # Initialize World
    world = FeederWorld()
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # Reset fitness for new epoch
        for f in world.fish:
            f.fitness = 0.0
            f.age = 0.0
            f.hunger = 0.0
            f.energy = 20.0
            f.alive = True
            
        # Run Simulation
        sim_time = 0.0
        steps = int(EPOCH_DURATION / DT)
        
        for _ in range(steps):
            world.update(DT)
            
        # Evaluate
        alive_fish = [f for f in world.fish if f.alive]
        if not alive_fish:
            print(f"Epoch {epoch}: EXTINCTION! Restarting population...")
            world = FeederWorld() # Hard reset
            continue
            
        fitnesses = [f.fitness for f in alive_fish]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        best_fish = max(alive_fish, key=lambda f: f.fitness)
        
        # Log
        logger.log(epoch, avg_fitness, max_fitness, len(alive_fish))
        logger.plot()
        
        print(f"Epoch {epoch}/{EPOCHS} | Avg Fit: {avg_fitness:.2f} | Max Fit: {max_fitness:.2f} | Pop: {len(alive_fish)} | Time: {time.time()-start_time:.2f}s")
        
        # Save Best Model
        model_path = MODEL_DIR / f"model_epoch_{epoch}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_fish.net, f)
            
        # Evolution (Simple Genetic Algorithm)
        # Keep top 20%
        alive_fish.sort(key=lambda f: f.fitness, reverse=True)
        survivors = alive_fish[:max(2, int(len(alive_fish) * 0.2))]
        
        # Repopulate
        new_fish = []
        while len(new_fish) < INITIAL_FISH_COUNT:
            parent = survivors[len(new_fish) % len(survivors)]
            child_net = parent.net.clone()
            child_net.mutate(0.05, 0.1)
            
            # Create new fish with this brain
            # We need to import FeederFish here or use world's method if exposed
            # But we can just instantiate FeederFish
            from feeder_entities import FeederFish
            import random
            from feeder_constants import WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH
            
            child = FeederFish(
                random.uniform(0, WORLD_WIDTH),
                random.uniform(0, WORLD_HEIGHT),
                random.uniform(0, WORLD_DEPTH),
                child_net
            )
            new_fish.append(child)
            
        world.fish = new_fish
        
    print("Training Complete!")

if __name__ == "__main__":
    train()
