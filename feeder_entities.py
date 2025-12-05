"""Entity classes for Bottom Feeder simulation."""
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List

# Import from Archive
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Archieve"))

from Archieve.neural_network import FeedForwardNet
from Archieve.utils import clamp

from feeder_constants import (
    WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH,
    FISH_MIN_SIZE, FISH_MAX_SIZE, FISH_MATURE_AGE,
    FISH_MIN_SPEED, FISH_MAX_SPEED,
    HUNGER_BASE_RATE, HUNGER_SIZE_FACTOR, HUNGER_SPEED_FACTOR,
    HUNGER_MAX, REPRO_ENERGY_COST, REPRO_MIN_ENERGY, REPRO_MIN_AGE,
    NUM_SENSORS
)

@dataclass
class FeederFood:
    x: float
    y: float
    z: float
    active: bool = True

class FeederFish:
    def __init__(self, x, y, z, net=None):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = random.uniform(0, math.tau)
        self.pitch = random.uniform(-math.pi/4, math.pi/4)
        self.speed = random.uniform(FISH_MIN_SPEED, FISH_MAX_SPEED)
        
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.age = 0.0
        self.size = FISH_MIN_SIZE
        self.hunger = 0.0
        self.energy = 20.0
        self.alive = True
        
        # Neural Network
        # Inputs: 
        # 0-4: Sensors (5)
        # 5: Speed (normalized)
        # 6: Hunger (normalized)
        # 7: Size (normalized)
        # 8: Energy (normalized)
        # 9: Last Yaw Rate
        # 10: Last Pitch Rate
        self.input_size = NUM_SENSORS + 6
        
        if net:
            self.net = net
        else:
            self.net = FeedForwardNet(self.input_size, [16, 12], 3) # 3 outputs: Yaw, Pitch, Speed
            
        self.sensors = [0.0] * NUM_SENSORS
        self.last_yaw_rate = 0.0
        self.last_pitch_rate = 0.0
        
        self.fitness = 0.0
        
    def update(self, dt):
        if not self.alive:
            return

        self.age += dt
        
        # Growth
        if self.age < FISH_MATURE_AGE:
            growth = self.age / FISH_MATURE_AGE
            self.size = FISH_MIN_SIZE + (FISH_MAX_SIZE - FISH_MIN_SIZE) * growth
        else:
            self.size = FISH_MAX_SIZE
            
        # Hunger Logic
        # Faster = Hungrier
        # Bigger = Hungrier
        hunger_rate = (HUNGER_BASE_RATE + 
                       self.size * HUNGER_SIZE_FACTOR + 
                       (self.speed / FISH_MAX_SPEED) * HUNGER_SPEED_FACTOR * 10.0)
        
        self.hunger += hunger_rate * dt
        
        if self.hunger >= HUNGER_MAX:
            self.alive = False
            return
            
        # Prepare Inputs
        inputs = np.zeros(self.input_size, dtype=np.float32)
        
        # Sensors (already normalized by physics engine logic mostly, but let's clamp)
        for i in range(NUM_SENSORS):
            inputs[i] = clamp(self.sensors[i], 0.0, 5.0) / 5.0 # Normalize somewhat
            
        # Internal State
        inputs[NUM_SENSORS] = (self.speed - FISH_MIN_SPEED) / (FISH_MAX_SPEED - FISH_MIN_SPEED) * 2 - 1
        inputs[NUM_SENSORS+1] = (self.hunger / HUNGER_MAX) * 2 - 1
        inputs[NUM_SENSORS+2] = (self.size - FISH_MIN_SIZE) / (FISH_MAX_SIZE - FISH_MIN_SIZE) * 2 - 1
        inputs[NUM_SENSORS+3] = clamp(self.energy / 100.0, 0.0, 1.0) * 2 - 1
        inputs[NUM_SENSORS+4] = self.last_yaw_rate
        inputs[NUM_SENSORS+5] = self.last_pitch_rate
        
        # NN Forward
        outputs = self.net.forward(inputs)
        
        # Interpret Outputs
        # 0: Yaw Rate (-1 to 1) -> Scale to max turn rate
        # 1: Pitch Rate (-1 to 1)
        # 2: Target Speed (-1 to 1)
        
        yaw_rate = outputs[0] * 3.0 # Rad/s
        pitch_rate = outputs[1] * 2.0
        target_speed = ((outputs[2] + 1.0) / 2.0) * (FISH_MAX_SPEED - FISH_MIN_SPEED) + FISH_MIN_SPEED
        
        self.last_yaw_rate = outputs[0]
        self.last_pitch_rate = outputs[1]
        
        # Apply Movement
        self.yaw += yaw_rate * dt
        self.pitch += pitch_rate * dt
        self.pitch = clamp(self.pitch, -math.pi/2 + 0.1, math.pi/2 - 0.1)
        
        # Smooth speed change
        self.speed += (target_speed - self.speed) * 2.0 * dt
        
        # Calculate Velocity
        xz = math.cos(self.pitch)
        self.vx = xz * math.cos(self.yaw) * self.speed
        self.vy = math.sin(self.pitch) * self.speed
        self.vz = xz * math.sin(self.yaw) * self.speed
        
        # Update Position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        # Boundaries (Bounce)
        if self.x < 0: self.x = 0; self.vx *= -1; self.yaw = math.pi - self.yaw
        if self.x > WORLD_WIDTH: self.x = WORLD_WIDTH; self.vx *= -1; self.yaw = math.pi - self.yaw
        
        if self.y < 0: self.y = 0; self.vy *= -1; self.pitch = -self.pitch
        if self.y > WORLD_HEIGHT: self.y = WORLD_HEIGHT; self.vy *= -1; self.pitch = -self.pitch
        
        if self.z < 0: self.z = 0; self.vz *= -1; self.yaw = -self.yaw
        if self.z > WORLD_DEPTH: self.z = WORLD_DEPTH; self.vz *= -1; self.yaw = -self.yaw
        
        # Fitness accumulation (survival)
        self.fitness += dt

    def eat(self, count, nutrition):
        if count <= 0: return
        
        energy_gain = count * nutrition * 5.0
        hunger_loss = count * nutrition * 2.0
        
        self.energy += energy_gain
        self.hunger = max(0.0, self.hunger - hunger_loss)
        self.fitness += count * 10.0 # Big reward for eating

    def can_reproduce(self):
        return (self.alive and 
                self.energy >= REPRO_MIN_ENERGY and 
                self.age >= REPRO_MIN_AGE)
