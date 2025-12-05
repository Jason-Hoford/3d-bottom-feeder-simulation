"""World management for Bottom Feeder simulation."""
import random
import numpy as np
import taichi as ti
from typing import List

from feeder_constants import (
    WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH,
    MAX_FOOD, INITIAL_FISH_COUNT,
    FOOD_RESPAWN_TIME, FOOD_NUTRITION,
    REPRO_ENERGY_COST
)
from feeder_entities import FeederFish
from feeder_physics import FeederPhysics

class FeederWorld:
    def __init__(self):
        self.physics = FeederPhysics(max_fish=1000) # Support up to 1000 fish
        self.fish: List[FeederFish] = []
        self.food_respawn_queue = [] # List of (index, respawn_time)
        self.sim_time = 0.0
        
        self._init_food()
        self._init_fish()
        
    def _init_food(self):
        # Generate random food positions
        positions = np.random.rand(MAX_FOOD, 3).astype(np.float32)
        positions[:, 0] *= WORLD_WIDTH
        positions[:, 1] *= WORLD_HEIGHT
        positions[:, 2] *= WORLD_DEPTH
        
        # Set to Taichi
        self.physics.food_pos.from_numpy(positions)
        self.physics.food_active.fill(1)
        
    def _init_fish(self):
        for _ in range(INITIAL_FISH_COUNT):
            self.fish.append(FeederFish(
                random.uniform(0, WORLD_WIDTH),
                random.uniform(0, WORLD_HEIGHT),
                random.uniform(0, WORLD_DEPTH)
            ))
            
    def update(self, dt):
        self.sim_time += dt
        
        # 1. Sync Fish to Taichi
        num_fish = len(self.fish)
        if num_fish > self.physics.max_fish:
            # Cap population or handle resizing (for now just cap logic in reproduction)
            num_fish = self.physics.max_fish
            self.fish = self.fish[:num_fish]
            
        fish_pos = np.zeros((self.physics.max_fish, 3), dtype=np.float32)
        fish_rot = np.zeros((self.physics.max_fish, 2), dtype=np.float32)
        fish_size = np.zeros(self.physics.max_fish, dtype=np.float32)
        fish_active = np.zeros(self.physics.max_fish, dtype=np.int32)
        
        for i, f in enumerate(self.fish):
            if f.alive:
                fish_pos[i] = [f.x, f.y, f.z]
                fish_rot[i] = [f.yaw, f.pitch]
                fish_size[i] = f.size
                fish_active[i] = 1
                
        self.physics.fish_pos.from_numpy(fish_pos)
        self.physics.fish_rot.from_numpy(fish_rot)
        self.physics.fish_size.from_numpy(fish_size)
        self.physics.fish_active.from_numpy(fish_active)
        
        # 2. Run Physics
        self.physics.update_sensors()
        self.physics.check_collisions()
        
        # 3. Read back results
        sensors = self.physics.fish_sensors.to_numpy()
        eaten_counts = self.physics.fish_eaten_count.to_numpy()
        
        # Check for eaten food to schedule respawn
        # This is the slow part if we scan everything.
        # Optimization: We know how many were eaten, but not WHICH ones easily without scanning.
        # However, check_collisions marks them inactive in Taichi.
        # We can pull `food_active` and check diff?
        # Or just randomly respawn inactive ones?
        # Let's do a periodic scan or just scan every frame if MAX_FOOD is 5000 (fast enough).
        
        food_active = self.physics.food_active.to_numpy()
        # Find indices where food became inactive but we haven't queued them yet?
        # Actually, we can just rebuild the queue from 0s in food_active that are NOT in queue?
        # Simpler: Just iterate all food_active. If 0, check if it's already in queue.
        # Even simpler: We don't track "which" food is in queue, just "a" food slot.
        # But we need to know which index to reactivate.
        
        # Let's just scan for 0s.
        inactive_indices = np.where(food_active == 0)[0]
        
        # We need to know which ones were JUST eaten this frame to avoid re-adding to queue.
        # But simpler logic:
        # If food_active[i] == 0, it means it's gone.
        # We can just immediately queue it for respawn if it's not already scheduled.
        # But checking "if not already scheduled" is O(N).
        # 
        # Alternative:
        # In Taichi, when we eat food, we return its index?
        # Taichi doesn't support dynamic lists return easily.
        # 
        # Let's use a set for `food_respawn_queue` indices.
        
        current_queued_indices = set(idx for idx, t in self.food_respawn_queue)
        newly_eaten = [idx for idx in inactive_indices if idx not in current_queued_indices]
        
        for idx in newly_eaten:
            self.food_respawn_queue.append((idx, self.sim_time + FOOD_RESPAWN_TIME))
            
        # 4. Update Fish Logic
        new_fish = []
        dead_fish_indices = []
        
        for i, f in enumerate(self.fish):
            if not f.alive:
                continue
                
            # Update sensors
            f.sensors = sensors[i]
            
            # Handle eating
            count = eaten_counts[i]
            if count > 0:
                f.eat(count, FOOD_NUTRITION)
                
            # Update logic
            f.update(dt)
            
            # Reproduction
            if f.can_reproduce() and len(self.fish) + len(new_fish) < self.physics.max_fish:
                f.energy -= REPRO_ENERGY_COST
                child_net = f.net.clone()
                child_net.mutate(0.05, 0.1)
                
                # Spawn child nearby
                child = FeederFish(f.x, f.y, f.z, child_net)
                new_fish.append(child)
                
        self.fish.extend(new_fish)
        
        # Cleanup dead fish
        self.fish = [f for f in self.fish if f.alive]
        
        # If population crashes, respawn some
        if len(self.fish) < 10:
             for _ in range(10):
                self.fish.append(FeederFish(
                    random.uniform(0, WORLD_WIDTH),
                    random.uniform(0, WORLD_HEIGHT),
                    random.uniform(0, WORLD_DEPTH)
                ))

        # 5. Respawn Food
        # Process queue
        remaining_queue = []
        respawn_indices = []
        respawn_positions = []
        
        for idx, respawn_time in self.food_respawn_queue:
            if self.sim_time >= respawn_time:
                respawn_indices.append(idx)
                respawn_positions.append([
                    random.uniform(0, WORLD_WIDTH),
                    random.uniform(0, WORLD_HEIGHT),
                    random.uniform(0, WORLD_DEPTH)
                ])
            else:
                remaining_queue.append((idx, respawn_time))
                
        self.food_respawn_queue = remaining_queue
        
        if respawn_indices:
            # Update Taichi
            # We need to write specific indices.
            # Taichi doesn't support scatter write from list easily in Python scope without loop.
            # But we can update the numpy array and re-upload? No, that overwrites everything.
            # We can use a kernel to respawn specific indices?
            # Or just loop in Python and write to field[i].
            
            for i, idx in enumerate(respawn_indices):
                self.physics.food_pos[idx] = respawn_positions[i]
                self.physics.food_active[idx] = 1

    def get_pending_food_count(self):
        return len(self.food_respawn_queue)

    def save_state(self, filename):
        import pickle
        # Gather state
        state = {
            "sim_time": self.sim_time,
            "fish": self.fish, # Fish are dataclasses/objects, pickleable
            "food_respawn_queue": self.food_respawn_queue,
            # Taichi fields need to be numpy
            "food_pos": self.physics.food_pos.to_numpy(),
            "food_active": self.physics.food_active.to_numpy()
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
            
    def load_state(self, filename):
        import pickle
        with open(filename, "rb") as f:
            state = pickle.load(f)
            
        self.sim_time = state["sim_time"]
        self.fish = state["fish"]
        self.food_respawn_queue = state["food_respawn_queue"]
        
        # Restore Taichi
        self.physics.food_pos.from_numpy(state["food_pos"])
        self.physics.food_active.from_numpy(state["food_active"])
        
        # Sync fish to Taichi immediately
        # (Will happen in next update(), but good to do now for rendering)
        # We can just let update() handle it, but we need to ensure physics arrays are sized right if fish count changed?
        # Physics max_fish is fixed at init. If loaded fish > max_fish, we might have issues.
        # But we capped it in update().

