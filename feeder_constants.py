"""Constants for the Bottom Feeder simulation."""
from pathlib import Path

# World dimensions
WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000
WORLD_DEPTH = 2000

# Viewport dimensions
VIEW_WIDTH = 1200
VIEW_HEIGHT = 900

# Simulation parameters
INITIAL_FISH_COUNT = 150
MAX_FOOD = 10000  # Increased density
FOOD_SIZE = 2.0  # Tiny particles
FOOD_NUTRITION = 1.0  # Low value per particle
FOOD_RESPAWN_TIME = 5.0 # Seconds before eaten food respawns

FPS = 60
SAVE_DIR = Path(".")
SAVE_PATTERN = "feeder_sim_*.pkl"

# Camera settings
CAMERA_ROTATION_SPEED = 0.003
CAMERA_MOVE_SPEED = 400.0
MOUSE_SENSITIVITY = 0.002
DEFAULT_ZOOM = 0.4
MIN_ZOOM = 0.1
MAX_ZOOM = 3.0

# Fish parameters
FISH_MIN_SIZE = 6.0
FISH_MAX_SIZE = 20.0
FISH_MATURE_AGE = 30.0
FISH_MAX_SPEED = 150.0
FISH_MIN_SPEED = 20.0

# Hunger & Energy
# Hunger grows based on size and speed
# hunger_rate = base + size * factor + speed * factor
HUNGER_BASE_RATE = 0.5  # Lowered from 1.0
HUNGER_SIZE_FACTOR = 0.1  # Lowered from 0.2
HUNGER_SPEED_FACTOR = 0.02  # Lowered from 0.05
HUNGER_MAX = 100.0

# Reproduction
REPRO_ENERGY_COST = 30.0  # Lowered from 50.0
REPRO_MIN_ENERGY = 40.0  # Lowered from 60.0
REPRO_MIN_AGE = 10.0  # Lowered from 15.0

# Sensing (FOV)
FOV_ANGLE = 120.0  # Degrees
SENSOR_RANGE = 400.0
NUM_SENSORS = 5  # Number of sectors in FOV
# 5 sensors: Far Left, Left, Center, Right, Far Right

# Physics / Taichi
GRID_SIZE = 50  # Spatial hashing grid size

# Graphs
HUNGER_GRAPH_MAX = 100.0
GRAPH_TIME_WINDOW = 20.0
MAX_CONSUMPTION_RATE = 200.0  # Increased for better scaling

# Visualization Modes
VIS_MODE_NORMAL = 0
VIS_MODE_DENSITY = 1
VIS_MODE_FLOW = 2

# Density Grid
DENSITY_GRID_RES = 30  # 30x30x30 grid for smoother gas visualization
