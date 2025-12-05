"""Renderer for Bottom Feeder simulation."""
import math
import pygame
import numpy as np
from typing import Tuple, Optional

from feeder_constants import (
    VIEW_WIDTH, VIEW_HEIGHT,
    WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH,
    FOOD_SIZE, VIS_MODE_NORMAL, VIS_MODE_DENSITY, VIS_MODE_FLOW,
    DENSITY_GRID_RES
)
from feeder_world import FeederWorld

# Utility vector functions
def v3_sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def v3_dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def v3_cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def v3_norm(a):
    l = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    if l < 1e-9: return (0,0,1)
    return (a[0]/l, a[1]/l, a[2]/l)

class Camera3D:
    def __init__(self):
        self.center = (WORLD_WIDTH/2, WORLD_HEIGHT/2, WORLD_DEPTH/2)
        self.distance = 2500.0
        self.angle_x = 0.5
        self.angle_y = 0.3
        self.zoom = 0.4
        self.fov = math.pi / 4.0
        self.near = 1.0
        self.width = VIEW_WIDTH
        self.height = VIEW_HEIGHT
        
    def get_pos(self):
        cx = self.distance * math.cos(self.angle_y) * math.sin(self.angle_x)
        cy = self.distance * math.sin(self.angle_y)
        cz = self.distance * math.cos(self.angle_y) * math.cos(self.angle_x)
        return (self.center[0]+cx, self.center[1]+cy, self.center[2]+cz)

    def project(self, p):
        cam_pos = self.get_pos()
        forward = v3_norm(v3_sub(self.center, cam_pos))
        up_world = (0,1,0)
        right = v3_norm(v3_cross(forward, up_world))
        up = v3_norm(v3_cross(right, forward))
        
        to_p = v3_sub(p, cam_pos)
        
        z = v3_dot(to_p, forward)
        if z < self.near: return None
        
        x = v3_dot(to_p, right)
        y = v3_dot(to_p, up)
        
        scale = (1.0 / math.tan(self.fov/2)) / z * self.zoom * self.height
        
        sx = int(self.width/2 + x * scale)
        sy = int(self.height/2 - y * scale)
        return (sx, sy, z)

# Import from Archive
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Archieve"))
from Archieve.utils import nn_to_color

def draw_feeder_world(screen: pygame.Surface, world: FeederWorld, camera: Camera3D, vis_mode: int = VIS_MODE_NORMAL):
    screen.fill((5, 5, 15)) # Dark deep ocean
    
    if vis_mode == VIS_MODE_DENSITY:
        draw_density_mode(screen, world, camera)
        return
    elif vis_mode == VIS_MODE_FLOW:
        draw_flow_mode(screen, world, camera)
        return
        
    # NORMAL MODE (Existing code)
    # 1. Get Data
    # Food
    food_pos = world.physics.food_pos.to_numpy()
    food_active = world.physics.food_active.to_numpy()
    
    # Filter active
    active_indices = np.where(food_active == 1)[0]
    active_food = food_pos[active_indices]
    
    # Fish
    fish_list = world.fish
    
    # 2. Project Everything
    drawables = []
    
    # Project Food
    cam_pos = camera.get_pos()
    cam_center = camera.center
    
    # Precompute camera vectors
    forward = v3_norm(v3_sub(cam_center, cam_pos))
    right = v3_norm(v3_cross(forward, (0,1,0)))
    up = v3_norm(v3_cross(right, forward))
    
    cx, cy, cz = cam_pos
    fx, fy, fz = forward
    rx, ry, rz = right
    ux, uy, uz = up
    
    # Vectorized Projection for Food
    if len(active_food) > 0:
        # P - C
        dx = active_food[:, 0] - cx
        dy = active_food[:, 1] - cy
        dz = active_food[:, 2] - cz
        
        # Dot products
        z_cam = dx*fx + dy*fy + dz*fz
        
        # Filter behind camera
        mask = z_cam > camera.near
        
        if np.any(mask):
            z_valid = z_cam[mask]
            dx_v = dx[mask]
            dy_v = dy[mask]
            dz_v = dz[mask]
            
            x_cam = dx_v*rx + dy_v*ry + dz_v*rz
            y_cam = dx_v*ux + dy_v*uy + dz_v*uz
            
            scale = (1.0 / math.tan(camera.fov/2)) / z_valid * camera.zoom * camera.height
            
            sx = (camera.width/2 + x_cam * scale).astype(int)
            sy = (camera.height/2 - y_cam * scale).astype(int)
            
            # Zip it up
            for i in range(len(z_valid)):
                drawables.append((z_valid[i], 0, sx[i], sy[i])) # 0 = food
                
    # Project Fish
    # We need to project the triangle points, not just the center
    for f in fish_list:
        if not f.alive: continue
        
        # Calculate fish orientation vectors
        # forward
        f_fx = math.cos(f.pitch) * math.cos(f.yaw)
        f_fy = math.sin(f.pitch)
        f_fz = math.cos(f.pitch) * math.sin(f.yaw)
        
        # right
        f_rx = -math.sin(f.yaw)
        f_rz = math.cos(f.yaw)
        
        # up
        f_ux = -math.sin(f.pitch) * math.cos(f.yaw)
        f_uy = math.cos(f.pitch)
        f_uz = -math.sin(f.pitch) * math.sin(f.yaw)
        
        size = f.size
        
        # Nose
        nose = (
            f.x + f_fx * size * 1.2,
            f.y + f_fy * size * 1.2,
            f.z + f_fz * size * 1.2
        )
        
        # Base points
        base_offset = -size * 0.4
        left = (
            f.x + f_fx * base_offset + f_rx * size * 0.7,
            f.y + f_fy * base_offset + f_uy * size * 0.4,
            f.z + f_fz * base_offset + f_rz * size * 0.7
        )
        right = (
            f.x + f_fx * base_offset - f_rx * size * 0.7,
            f.y + f_fy * base_offset - f_uy * size * 0.4,
            f.z + f_fz * base_offset - f_rz * size * 0.7
        )
        
        p_nose = camera.project(nose)
        p_left = camera.project(left)
        p_right = camera.project(right)
        
        if p_nose and p_left and p_right:
            # Average depth
            avg_z = (p_nose[2] + p_left[2] + p_right[2]) / 3.0
            drawables.append((avg_z, 1, f, p_nose, p_left, p_right)) # 1 = fish

    # 3. Sort
    drawables.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Draw
    for item in drawables:
        if item[1] == 0:
            # Food
            _, _, sx, sy = item
            depth = item[0]
            alpha = max(50, min(255, int(255 - depth * 0.05)))
            color = (alpha, alpha, alpha)
            if 0 <= sx < VIEW_WIDTH and 0 <= sy < VIEW_HEIGHT:
                screen.set_at((sx, sy), color)
                if depth < 500:
                    screen.set_at((sx+1, sy), color)
                    screen.set_at((sx, sy+1), color)
                    screen.set_at((sx+1, sy+1), color)
        else:
            # Fish
            _, _, fish, p1, p2, p3 = item
            draw_fish_triangle(screen, fish, p1, p2, p3, item[0])

    # 5. Draw Borders
    draw_borders(screen, camera)

def draw_density_mode(screen, world, camera):
    # Retrieve density grid from Taichi
    grid = world.physics.density_grid.to_numpy()
    
    cell_w = WORLD_WIDTH / DENSITY_GRID_RES
    cell_h = WORLD_HEIGHT / DENSITY_GRID_RES
    cell_d = WORLD_DEPTH / DENSITY_GRID_RES
    
    # OPTIMIZATION: Only process non-zero cells
    indices = np.argwhere(grid > 0)
    
    if len(indices) == 0:
        draw_borders(screen, camera)
        return
    
    max_count = np.max(grid)
    cells_to_draw = []
    
    # Quick projection of only active cells
    for ix, iy, iz in indices:
        count = grid[ix, iy, iz]
        
        # Center of cell
        cx = (ix + 0.5) * cell_w
        cy = (iy + 0.5) * cell_h
        cz = (iz + 0.5) * cell_d
        
        proj = camera.project((cx, cy, cz))
        if proj:
            sx, sy, z = proj
            cells_to_draw.append((z, count, sx, sy))
    
    # Sort back-to-front
    cells_to_draw.sort(key=lambda x: x[0], reverse=True)
    
    # Pre-create cloud surfaces (cache common sizes)
    cloud_cache = {}
    
    # Draw volumetric clouds
    for z, count, sx, sy in cells_to_draw:
        # Normalize intensity
        intensity = min(1.0, count / (max_count * 0.3))
        
        # Heatmap color
        if intensity < 0.25:
            t = intensity / 0.25
            r, g, b = 0, int(150 * t), int(255 * (1 - t * 0.3))
        elif intensity < 0.5:
            t = (intensity - 0.25) / 0.25
            r, g, b = 0, int(150 + 105 * t), int(180 * (1 - t))
        elif intensity < 0.75:
            t = (intensity - 0.5) / 0.25
            r, g, b = int(255 * t), 255, 0
        else:
            t = (intensity - 0.75) / 0.25
            r, g, b = 255, int(255 * (1 - t)), 0
        
        # Lower alpha for gas effect
        alpha = int(min(150, 50 + intensity * 100))
        
        # Size based on depth and intensity
        base_size = max(15, int(2500 / z * (0.4 + 0.6 * intensity)))
        
        # Draw optimized cloud
        draw_fast_cloud(screen, sx, sy, base_size, (r, g, b), alpha, cloud_cache)
    
    draw_borders(screen, camera)

def draw_fast_cloud(screen, cx, cy, size, color, alpha, cache):
    """Optimized cloud rendering with caching"""
    # Round size for caching
    cache_size = (size // 5) * 5
    cache_key = (cache_size, alpha)
    
    # Check cache
    if cache_key not in cache:
        surf_size = int(cache_size * 2.5)
        if surf_size < 4:
            return
        
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = surf_size // 2
        
        # Simplified 3-layer gradient for speed
        for i in range(3):
            layer_size = cache_size * (1.1 - i * 0.35)
            layer_alpha = int(alpha * (1.0 - i * 0.6))
            
            if layer_size < 1:
                continue
            
            # Draw white circle on temp surface, set alpha globally
            temp_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (255, 255, 255), (center, center), int(layer_size))
            temp_surf.set_alpha(layer_alpha)
            surf.blit(temp_surf, (0, 0))
        
        cache[cache_key] = (surf, center)
    
    surf, center = cache[cache_key]
    
    # Tint the cached surface with the color
    tinted = surf.copy()
    tinted.fill(color + (255,), special_flags=pygame.BLEND_RGBA_MULT)
    
    screen.blit(tinted, (cx - center, cy - center), special_flags=pygame.BLEND_ALPHA_SDL2)


def draw_flow_mode(screen, world, camera):
    """Draw connected hierarchical tree showing flow from fine details to main trunks"""
    fish_list = world.fish
    
    # Define 4 grid levels (finest to coarsest)
    grid_levels = [
        {'res': 16, 'smooth': 0.1, 'min_fish': 2},    # Finest - need at least 2 fish
        {'res': 8, 'smooth': 0.3, 'min_fish': 4},     # Medium - need at least 4 fish
        {'res': 4, 'smooth': 0.6, 'min_fish': 8},     # Large - need at least 8 fish
        {'res': 2, 'smooth': 0.85, 'min_fish': 15},   # Main trunks - need at least 15 fish
    ]
    
    # Initialize smoothed velocity storage if not exists
    if not hasattr(draw_flow_mode, 'smoothed_grids'):
        draw_flow_mode.smoothed_grids = [{} for _ in grid_levels]
    
    # Compute velocity grids for all levels
    level_grids = []
    
    for level_idx, level in enumerate(grid_levels):
        grid_res = level['res']
        cell_w = WORLD_WIDTH / grid_res
        cell_h = WORLD_HEIGHT / grid_res
        cell_d = WORLD_DEPTH / grid_res
        
        velocity_grid = {}
        
        for f in fish_list:
            if not f.alive:
                continue
            
            ix = int(f.x / cell_w)
            iy = int(f.y / cell_h)
            iz = int(f.z / cell_d)
            
            ix = max(0, min(ix, grid_res - 1))
            iy = max(0, min(iy, grid_res - 1))
            iz = max(0, min(iz, grid_res - 1))
            
            vx = math.cos(f.pitch) * math.cos(f.yaw) * f.speed
            vy = math.sin(f.pitch) * f.speed
            vz = math.cos(f.pitch) * math.sin(f.yaw) * f.speed
            
            key = (ix, iy, iz)
            if key not in velocity_grid:
                velocity_grid[key] = [0.0, 0.0, 0.0, 0]
            
            velocity_grid[key][0] += vx
            velocity_grid[key][1] += vy
            velocity_grid[key][2] += vz
            velocity_grid[key][3] += 1
        
        # Apply temporal smoothing (larger branches move slower)
        smooth_factor = level['smooth']
        smoothed = draw_flow_mode.smoothed_grids[level_idx]
        
        # Update smoothed values
        new_smoothed = {}
        
        for key, (vx, vy, vz, count) in velocity_grid.items():
            if key in smoothed:
                old_vx, old_vy, old_vz, old_count = smoothed[key]
                new_smoothed[key] = [
                    smooth_factor * old_vx + (1 - smooth_factor) * vx,
                    smooth_factor * old_vy + (1 - smooth_factor) * vy,
                    smooth_factor * old_vz + (1 - smooth_factor) * vz,
                    smooth_factor * old_count + (1 - smooth_factor) * count
                ]
            else:
                new_smoothed[key] = [vx, vy, vz, count]
        
        # Decay old cells
        for key in smoothed:
            if key not in velocity_grid:
                old_vx, old_vy, old_vz, old_count = smoothed[key]
                decayed_count = old_count * smooth_factor
                if decayed_count > 0.1:
                    new_smoothed[key] = [
                        old_vx * smooth_factor,
                        old_vy * smooth_factor,
                        old_vz * smooth_factor,
                        decayed_count
                    ]
        
        draw_flow_mode.smoothed_grids[level_idx] = new_smoothed
        
        level_grids.append({
            'grid': new_smoothed,
            'res': grid_res,
            'cell_w': cell_w,
            'cell_h': cell_h,
            'cell_d': cell_d,
            'level': level_idx,
            'min_fish': level['min_fish']
        })
    
    # Calculate max fish count per parent for relative strength
    parent_max_counts = {}
    for level_idx in range(len(level_grids)):
        if level_idx >= len(level_grids) - 1:
            continue
        
        current = level_grids[level_idx]
        parent = level_grids[level_idx + 1]
        
        for (ix, iy, iz), (_, _, _, count) in current['grid'].items():
            if count < current['min_fish']:
                continue
            
            # Find parent cell
            cx = (ix + 0.5) * current['cell_w']
            cy = (iy + 0.5) * current['cell_h']
            cz = (iz + 0.5) * current['cell_d']
            
            parent_ix = int(cx / parent['cell_w'])
            parent_iy = int(cy / parent['cell_h'])
            parent_iz = int(cz / parent['cell_d'])
            
            parent_key = (level_idx, parent_ix, parent_iy, parent_iz)
            
            if parent_key not in parent_max_counts:
                parent_max_counts[parent_key] = count
            else:
                parent_max_counts[parent_key] = max(parent_max_counts[parent_key], count)
    
    # Draw connections from fine to coarse
    all_lines = []
    
    for level_idx in range(len(level_grids)):
        current = level_grids[level_idx]
        
        for (ix, iy, iz), (sum_vx, sum_vy, sum_vz, count) in current['grid'].items():
            # Minimum fish threshold
            if count < current['min_fish']:
                continue
            
            avg_vx = sum_vx / max(count, 1)
            avg_vy = sum_vy / max(count, 1)
            avg_vz = sum_vz / max(count, 1)
            speed = math.sqrt(avg_vx*avg_vx + avg_vy*avg_vy + avg_vz*avg_vz)
            
            if speed < 1.0:
                continue
            
            # Cell center (start point)
            cx = (ix + 0.5) * current['cell_w']
            cy = (iy + 0.5) * current['cell_h']
            cz = (iz + 0.5) * current['cell_d']
            
            # Calculate relative strength (compared to siblings)
            relative_strength = 1.0
            if level_idx < len(level_grids) - 1:
                parent = level_grids[level_idx + 1]
                parent_ix = int(cx / parent['cell_w'])
                parent_iy = int(cy / parent['cell_h'])
                parent_iz = int(cz / parent['cell_d'])
                parent_key = (level_idx, parent_ix, parent_iy, parent_iz)
                
                if parent_key in parent_max_counts:
                    max_sibling_count = parent_max_counts[parent_key]
                    if max_sibling_count > 0:
                        relative_strength = count / max_sibling_count
            
            # If not the coarsest level, connect directly to parent cell center
            if level_idx < len(level_grids) - 1:
                parent = level_grids[level_idx + 1]
                
                parent_ix = max(0, min(parent_ix, parent['res'] - 1))
                parent_iy = max(0, min(parent_iy, parent['res'] - 1))
                parent_iz = max(0, min(parent_iz, parent['res'] - 1))
                
                # Parent cell center (end point)
                end_x = (parent_ix + 0.5) * parent['cell_w']
                end_y = (parent_iy + 0.5) * parent['cell_h']
                end_z = (parent_iz + 0.5) * parent['cell_d']
            else:
                # Coarsest level - draw in flow direction
                norm = math.sqrt(avg_vx*avg_vx + avg_vy*avg_vy + avg_vz*avg_vz)
                if norm < 0.1:
                    continue
                
                dir_x = avg_vx / norm
                dir_y = avg_vy / norm
                dir_z = avg_vz / norm
                
                line_len = min(current['cell_w'], current['cell_h'], current['cell_d']) * 0.8
                end_x = cx + dir_x * line_len
                end_y = cy + dir_y * line_len
                end_z = cz + dir_z * line_len
                
                end_x = max(0, min(WORLD_WIDTH, end_x))
                end_y = max(0, min(WORLD_HEIGHT, end_y))
                end_z = max(0, min(WORLD_DEPTH, end_z))
            
            # Project
            proj_start = camera.project((cx, cy, cz))
            proj_end = camera.project((end_x, end_y, end_z))
            
            if not proj_start or not proj_end:
                continue
            
            # Thickness
            if level_idx == 3:
                base_thickness = 8
                thickness = base_thickness + min(10, int(count // 4))
            elif level_idx == 2:
                base_thickness = 5
                thickness = base_thickness + min(6, int(count // 5))
            elif level_idx == 1:
                base_thickness = 2
                thickness = base_thickness + min(4, int(count // 8))
            else:
                base_thickness = 1
                thickness = base_thickness + min(2, int(count // 10))
            
            # Base color gradient
            t = level_idx / (len(level_grids) - 1)
            
            if t < 0.25:
                s = t / 0.25
                r, g, b = 0, int(100 + 155 * s), 255
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                r, g, b = 0, 255, int(255 * (1 - s))
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                r, g, b = int(255 * s), 255, 0
            else:
                s = (t - 0.75) / 0.25
                r, g, b = 255, int(255 * (1 - s)), 0
            
            # Adjust brightness based on relative strength
            # Stronger flow = brighter, weaker flow = dimmer
            brightness = 0.4 + 0.6 * relative_strength  # 40% to 100%
            r = int(r * brightness)
            g = int(g * brightness)
            b = int(b * brightness)
            
            z = (proj_start[2] + proj_end[2]) / 2
            
            all_lines.append((
                level_idx,
                z,
                proj_start,
                proj_end,
                (r, g, b),
                thickness
            ))
    
    # Sort: coarse levels first, then by depth
    all_lines.sort(key=lambda x: (x[0], -x[1]))
    
    # Draw all connected lines
    for level_idx, z, p_start, p_end, color, thickness in all_lines:
        start = (p_start[0], p_start[1])
        end = (p_end[0], p_end[1])
        
        pygame.draw.line(screen, color, start, end, thickness)
    
    draw_borders(screen, camera)





def draw_simple_arrowhead(screen, start, end, color, size):
    """Draw a simple arrowhead"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length < 3:
        return
    
    # Normalize
    dx /= length
    dy /= length
    
    # Arrowhead dimensions
    arrow_len = min(size, length * 0.3)
    arrow_width = arrow_len * 0.5
    
    # Perpendicular
    px = -dy
    py = dx
    
    # Points
    tip = end
    left = (end[0] - dx * arrow_len + px * arrow_width, end[1] - dy * arrow_len + py * arrow_width)
    right = (end[0] - dx * arrow_len - px * arrow_width, end[1] - dy * arrow_len - py * arrow_width)
    
    # Draw
    pygame.draw.polygon(screen, color, [tip, left, right])




def draw_borders(screen, camera):
    border_color = (100, 120, 180)
    corners = [
        (0, 0, 0),
        (WORLD_WIDTH, 0, 0),
        (WORLD_WIDTH, 0, WORLD_DEPTH),
        (0, 0, WORLD_DEPTH),
        (0, WORLD_HEIGHT, 0),
        (WORLD_WIDTH, WORLD_HEIGHT, 0),
        (WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH),
        (0, WORLD_HEIGHT, WORLD_DEPTH),
    ]
    
    # Project corners
    projected_corners = []
    for corner in corners:
        proj = camera.project(corner)
        if proj:
            projected_corners.append((proj[0], proj[1]))
        else:
            projected_corners.append(None)
    
    # Draw edges
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    for edge in edges:
        p1_idx, p2_idx = edge
        if (p1_idx < len(projected_corners) and p2_idx < len(projected_corners) and
            projected_corners[p1_idx] is not None and projected_corners[p2_idx] is not None):
            p1 = projected_corners[p1_idx]
            p2 = projected_corners[p2_idx]
            pygame.draw.line(screen, border_color, p1, p2, 2)

def draw_fish_triangle(screen, fish, p1, p2, p3, depth):
    color = nn_to_color(fish.net.weights)
    
    # Fade based on depth
    depth_factor = min(1.0, max(0.3, 1.0 - (depth - 1.0) / 3000.0))
    faded_color = tuple(int(c * depth_factor) for c in color)
    
    points = [(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])]
    
    pygame.draw.polygon(screen, faded_color, points)
    # Outline
    outline_color = tuple(min(255, c + 40) for c in faded_color)
    pygame.draw.polygon(screen, outline_color, points, 1)

from feeder_constants import HUNGER_GRAPH_MAX, GRAPH_TIME_WINDOW, MAX_CONSUMPTION_RATE

def draw_graphs(
        screen: pygame.Surface,
        hunger_history: list[tuple[float, float]],
        consumption_history: list[tuple[float, float]],
        sim_time: float,
        show_graph: bool,
):
    if not show_graph:
        return

    width = 260
    hunger_height = 80
    consumption_height = 100
    spacing = 12
    margin = 10

    total_height = hunger_height + consumption_height + spacing
    x0 = VIEW_WIDTH - width - margin
    y0 = VIEW_HEIGHT - total_height - margin

    hunger_rect = pygame.Rect(x0, y0, width, hunger_height)
    consumption_rect = pygame.Rect(x0, y0 + hunger_height + spacing, width, consumption_height)

    font = pygame.font.SysFont("Arial", 14)

    draw_history_graph(
        screen,
        hunger_rect,
        hunger_history,
        sim_time,
        HUNGER_GRAPH_MAX,
        "Avg Hunger",
        font,
        y_step=20,
        color=(255, 160, 90),
        smooth_alpha=0.15,
    )

    draw_history_graph(
        screen,
        consumption_rect,
        consumption_history,
        sim_time,
        MAX_CONSUMPTION_RATE,
        "Food/Sec",
        font,
        y_step=10,
        color=(80, 200, 255),
        smooth_alpha=0.3,
    )

def draw_history_graph(
        screen: pygame.Surface,
        rect: pygame.Rect,
        history: list[tuple[float, float]],
        sim_time: float,
        max_value: float,
        label: str,
        font: pygame.font.Font,
        y_step: float,
        color: Tuple[int, int, int],
        smooth_alpha: float = 0.0,
        time_window: float = GRAPH_TIME_WINDOW,
):
    pygame.draw.rect(screen, (0, 0, 0), rect)
    pygame.draw.rect(screen, (80, 80, 80), rect, 1)

    recent = [(t, v) for (t, v) in history if sim_time - t <= time_window]
    axis_color = (120, 120, 120)
    grid_color = (40, 40, 70)

    # Axes
    pygame.draw.line(screen, axis_color, (rect.x + 30, rect.y + 10), (rect.x + 30, rect.bottom - 20), 1)
    pygame.draw.line(
        screen,
        axis_color,
        (rect.x + 30, rect.bottom - 20),
        (rect.right - 10, rect.bottom - 20),
        1,
    )

    # Grid lines (vertical every 2s)
    for step in range(0, int(time_window) + 1, 5):
        frac = step / time_window if time_window else 0.0
        gx = rect.x + 30 + frac * (rect.width - 40)
        pygame.draw.line(screen, grid_color, (gx, rect.y + 10), (gx, rect.bottom - 20), 1)

    # Horizontal grid
    if y_step > 0:
        current = 0.0
        while current <= max_value:
            frac = current / max_value if max_value else 0.0
            gy = rect.bottom - 20 - frac * (rect.height - 30)
            pygame.draw.line(screen, grid_color, (rect.x + 30, gy), (rect.right - 10, gy), 1)
            current += y_step

    label_surface = font.render(label, True, (200, 200, 200))
    screen.blit(label_surface, (rect.x + 5, rect.y + 5))

    if len(recent) < 2:
        return

    min_time = max(sim_time - time_window, recent[0][0])
    smooth_val = min(recent[0][1], max_value)
    smoothed: list[float] = []
    for _, value in recent:
        value = min(value, max_value)
        if smooth_alpha > 0:
            smooth_val = (1 - smooth_alpha) * smooth_val + smooth_alpha * value
            smoothed.append(smooth_val)
        else:
            smoothed.append(value)

    points = []
    for (t, _), smooth_value in zip(recent, smoothed):
        x_norm = (t - min_time) / time_window if time_window else 0.0
        y_norm = smooth_value / max_value if max_value else 0.0
        px = rect.x + 30 + x_norm * (rect.width - 40)
        py = rect.bottom - 20 - y_norm * (rect.height - 30)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.aalines(screen, color, False, points, blend=1)

def draw_hud(screen, world, fps):
    font = pygame.font.SysFont("Arial", 16)
    lines = [
        f"FPS: {fps:.1f}",
        f"Fish: {len(world.fish)}",
        f"Food: {np.sum(world.physics.food_active.to_numpy())}",
        f"Pending Respawn: {world.get_pending_food_count()}",
        f"Time: {world.sim_time:.1f}s"
    ]
    
    y = 10
    for line in lines:
        surf = font.render(line, True, (200, 200, 200))
        screen.blit(surf, (10, y))
        y += 20
