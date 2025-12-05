import taichi as ti
import math
from feeder_constants import (
    MAX_FOOD,
    INITIAL_FISH_COUNT,
    WORLD_WIDTH,
    WORLD_HEIGHT,
    WORLD_DEPTH,
    FOV_ANGLE,
    SENSOR_RANGE,
    NUM_SENSORS,
    FOOD_SIZE
)

@ti.data_oriented
class FeederPhysics:
    def __init__(self, max_fish=500):
        self.max_fish = max_fish
        
        # Fields
        self.food_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FOOD)
        self.food_active = ti.field(dtype=ti.i32, shape=MAX_FOOD)
        
        self.fish_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_fish)
        self.fish_rot = ti.Vector.field(2, dtype=ti.f32, shape=max_fish) # yaw, pitch
        self.fish_size = ti.field(dtype=ti.f32, shape=max_fish)
        self.fish_active = ti.field(dtype=ti.i32, shape=max_fish)
        
        # Sensor output: [fish_id, sensor_idx]
        self.fish_sensors = ti.field(dtype=ti.f32, shape=(max_fish, NUM_SENSORS))
        
        # Collision output: [fish_id] -> count of food eaten
        self.fish_eaten_count = ti.field(dtype=ti.i32, shape=max_fish)
        
        # Track which food was eaten by which fish to handle respawning in Python
        # This is a bit tricky to pass back efficiently. 
        # For now, we just mark food inactive in Taichi and let Python scan for inactive food if needed,
        # or we just rely on the count for the fish logic and handle food respawning by scanning `food_active` occasionally.
        
        self.fov_rad = math.radians(FOV_ANGLE)
        self.half_fov = self.fov_rad / 2.0
        self.sector_angle = self.fov_rad / NUM_SENSORS

    @ti.func
    def get_forward(self, yaw, pitch):
        # Convert yaw/pitch to forward vector
        # Consistent with entities.py:
        # vx = cos(pitch) * cos(yaw)
        # vy = sin(pitch)
        # vz = cos(pitch) * sin(yaw)
        xz_len = ti.cos(pitch)
        return ti.Vector([
            xz_len * ti.cos(yaw),
            ti.sin(pitch),
            xz_len * ti.sin(yaw)
        ])

    @ti.kernel
    def update_sensors(self):
        # Clear sensors
        for i, j in self.fish_sensors:
            self.fish_sensors[i, j] = 0.0

        for i in range(self.max_fish):
            if self.fish_active[i] == 0:
                continue
                
            pos = self.fish_pos[i]
            yaw = self.fish_rot[i][0]
            pitch = self.fish_rot[i][1]
            forward = self.get_forward(yaw, pitch)
            
            # Iterate all food
            for j in range(MAX_FOOD):
                if self.food_active[j] == 0:
                    continue
                    
                food_p = self.food_pos[j]
                diff = food_p - pos
                dist = diff.norm()
                
                if dist < SENSOR_RANGE and dist > 0.1:
                    # Check angle
                    # Project diff onto forward vector? 
                    # Actually we need the angle between forward and diff
                    direction = diff / dist
                    dot = forward.dot(direction)
                    # dot = cos(angle)
                    # angle = acos(dot)
                    
                    # Numerical stability clamp
                    if dot > 1.0: dot = 1.0
                    if dot < -1.0: dot = -1.0
                    
                    angle = ti.acos(dot)
                    
                    if angle < self.half_fov:
                        # It is within FOV. Now determine which sector.
                        # We need signed angle to determine left/right?
                        # In 3D, "left/right" is relative to the "up" vector of the fish.
                        # For simplicity, let's project to 2D (yaw plane) or just use the angle 
                        # relative to the center.
                        # 
                        # Actually, a simple way to map 3D FOV to 1D array of sensors is:
                        # We want to know if it's to the "left" or "right" of the forward vector.
                        # We can use the cross product with the up vector to find the "right" vector.
                        
                        # Up vector (approximate global up (0,1,0) or local up)
                        # Let's use global up (0,1,0) for simplicity, but fish pitch might mess this up.
                        # Better: calculate local right vector.
                        # right = cross(forward, global_up)
                        # If forward is straight up/down, this is unstable, but fish usually don't do that.
                        
                        global_up = ti.Vector([0.0, 1.0, 0.0])
                        right = forward.cross(global_up)
                        # If forward is parallel to up, right is zero. Handle this?
                        if right.norm_sqr() < 0.001:
                            right = ti.Vector([1.0, 0.0, 0.0]) # Fallback
                        else:
                            right = right.normalized()
                            
                        # Project direction onto right vector to get lateral offset
                        lateral = direction.dot(right)
                        
                        # We can map the angle range [-half_fov, half_fov] to [0, NUM_SENSORS]
                        # But acos always returns [0, pi]. It doesn't give sign.
                        # We use the lateral dot product to determine sign.
                        
                        signed_angle = angle
                        if lateral < 0:
                            signed_angle = -angle
                            
                        # Map [-half_fov, half_fov] -> [0, NUM_SENSORS]
                        # t goes from 0 to 1
                        t = (signed_angle + self.half_fov) / self.fov_rad
                        sensor_idx = int(t * NUM_SENSORS)
                        
                        # Clamp index just in case
                        if sensor_idx < 0: sensor_idx = 0
                        if sensor_idx >= NUM_SENSORS: sensor_idx = NUM_SENSORS - 1
import taichi as ti
import math
from feeder_constants import (
    MAX_FOOD,
    INITIAL_FISH_COUNT,
    WORLD_WIDTH,
    WORLD_HEIGHT,
    WORLD_DEPTH,
    FOV_ANGLE,
    SENSOR_RANGE,
    NUM_SENSORS,
    FOOD_SIZE,
    DENSITY_GRID_RES
)

@ti.data_oriented
class FeederPhysics:
    def __init__(self, max_fish=500):
        self.max_fish = max_fish
        
        # Fields
        self.food_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FOOD)
        self.food_active = ti.field(dtype=ti.i32, shape=MAX_FOOD)
        
        self.fish_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_fish)
        self.fish_rot = ti.Vector.field(2, dtype=ti.f32, shape=max_fish) # yaw, pitch
        self.fish_size = ti.field(dtype=ti.f32, shape=max_fish)
        self.fish_active = ti.field(dtype=ti.i32, shape=max_fish)
        
        # Sensor output: [fish_id, sensor_idx]
        self.fish_sensors = ti.field(dtype=ti.f32, shape=(max_fish, NUM_SENSORS))
        
        # Collision output: [fish_id] -> count of food eaten
        self.fish_eaten_count = ti.field(dtype=ti.i32, shape=max_fish)
        
        # Density Grid for Visualization
        self.density_grid = ti.field(dtype=ti.i32, shape=(DENSITY_GRID_RES, DENSITY_GRID_RES, DENSITY_GRID_RES))
        
        # Track which food was eaten by which fish to handle respawning in Python
        # This is a bit tricky to pass back efficiently. 
        # For now, we just mark food inactive in Taichi and let Python scan for inactive food if needed,
        # or we just rely on the count for the fish logic and handle food respawning by scanning `food_active` occasionally.
        
        self.fov_rad = math.radians(FOV_ANGLE)
        self.half_fov = self.fov_rad / 2.0
        self.sector_angle = self.fov_rad / NUM_SENSORS

    @ti.func
    def get_forward(self, yaw, pitch):
        # Convert yaw/pitch to forward vector
        # Consistent with entities.py:
        # vx = cos(pitch) * cos(yaw)
        # vy = sin(pitch)
        # vz = cos(pitch) * sin(yaw)
        xz_len = ti.cos(pitch)
        return ti.Vector([
            xz_len * ti.cos(yaw),
            ti.sin(pitch),
            xz_len * ti.sin(yaw)
        ])

    @ti.kernel
    def update_sensors(self):
        # Clear sensors
        for i, j in self.fish_sensors:
            self.fish_sensors[i, j] = 0.0

        for i in range(self.max_fish):
            if self.fish_active[i] == 0:
                continue
                
            pos = self.fish_pos[i]
            yaw = self.fish_rot[i][0]
            pitch = self.fish_rot[i][1]
            forward = self.get_forward(yaw, pitch)
            
            # Iterate all food
            for j in range(MAX_FOOD):
                if self.food_active[j] == 0:
                    continue
                    
                food_p = self.food_pos[j]
                diff = food_p - pos
                dist = diff.norm()
                
                if dist < SENSOR_RANGE and dist > 0.1:
                    # Check angle
                    # Project diff onto forward vector? 
                    # Actually we need the angle between forward and diff
                    direction = diff / dist
                    dot = forward.dot(direction)
                    # dot = cos(angle)
                    # angle = acos(dot)
                    
                    # Numerical stability clamp
                    if dot > 1.0: dot = 1.0
                    if dot < -1.0: dot = -1.0
                    
                    angle = ti.acos(dot)
                    
                    if angle < self.half_fov:
                        # It is within FOV. Now determine which sector.
                        # We need signed angle to determine left/right?
                        # In 3D, "left/right" is relative to the "up" vector of the fish.
                        # For simplicity, let's project to 2D (yaw plane) or just use the angle 
                        # relative to the center.
                        # 
                        # Actually, a simple way to map 3D FOV to 1D array of sensors is:
                        # We want to know if it's to the "left" or "right" of the forward vector.
                        # We can use the cross product with the up vector to find the "right" vector.
                        
                        # Up vector (approximate global up (0,1,0) or local up)
                        # Let's use global up (0,1,0) for simplicity, but fish pitch might mess this up.
                        # Better: calculate local right vector.
                        # right = cross(forward, global_up)
                        # If forward is straight up/down, this is unstable, but fish usually don't do that.
                        
                        global_up = ti.Vector([0.0, 1.0, 0.0])
                        right = forward.cross(global_up)
                        # If forward is parallel to up, right is zero. Handle this?
                        if right.norm_sqr() < 0.001:
                            right = ti.Vector([1.0, 0.0, 0.0]) # Fallback
                        else:
                            right = right.normalized()
                            
                        # Project direction onto right vector to get lateral offset
                        lateral = direction.dot(right)
                        
                        # We can map the angle range [-half_fov, half_fov] to [0, NUM_SENSORS]
                        # But acos always returns [0, pi]. It doesn't give sign.
                        # We use the lateral dot product to determine sign.
                        
                        signed_angle = angle
                        if lateral < 0:
                            signed_angle = -angle
                            
                        # Map [-half_fov, half_fov] -> [0, NUM_SENSORS]
                        # t goes from 0 to 1
                        t = (signed_angle + self.half_fov) / self.fov_rad
                        sensor_idx = int(t * NUM_SENSORS)
                        
                        # Clamp index just in case
                        if sensor_idx < 0: sensor_idx = 0
                        if sensor_idx >= NUM_SENSORS: sensor_idx = NUM_SENSORS - 1
                        
                        # Add density (1/dist squared or linear?)
                        # Linear density: (Range - Dist) / Range
                        density = (SENSOR_RANGE - dist) / SENSOR_RANGE
                        self.fish_sensors[i, sensor_idx] += density

    @ti.kernel
    def update_density_grid(self):
        # Clear grid
        for I in ti.grouped(self.density_grid):
            self.density_grid[I] = 0
            
        # Fill grid
        for i in range(MAX_FOOD):
            if self.food_active[i] == 1:
                pos = self.food_pos[i]
                # Map pos to grid index
                # World is 0 to WORLD_WIDTH etc.
                ix = int(pos.x / WORLD_WIDTH * DENSITY_GRID_RES)
                iy = int(pos.y / WORLD_HEIGHT * DENSITY_GRID_RES)
                iz = int(pos.z / WORLD_DEPTH * DENSITY_GRID_RES)
                
                # Clamp
                ix = ti.max(0, ti.min(ix, DENSITY_GRID_RES - 1))
                iy = ti.max(0, ti.min(iy, DENSITY_GRID_RES - 1))
                iz = ti.max(0, ti.min(iz, DENSITY_GRID_RES - 1))
                
                self.density_grid[ix, iy, iz] += 1

    @ti.kernel
    def check_collisions(self):
        # Clear eaten count
        for i in range(self.max_fish):
            self.fish_eaten_count[i] = 0
            
        for i in range(self.max_fish):
            if self.fish_active[i] == 0:
                continue
                
            pos = self.fish_pos[i]
            radius = self.fish_size[i]
            eat_radius_sq = (radius + FOOD_SIZE) ** 2
            
            for j in range(MAX_FOOD):
                if self.food_active[j] == 0:
                    continue
                    
                food_p = self.food_pos[j]
                diff = food_p - pos
                dist_sq = diff.norm_sqr()
                
                if dist_sq < eat_radius_sq:
                    # Eaten!
                    self.food_active[j] = 0
                    self.fish_eaten_count[i] += 1
