"""
Entidades del juego (Entities).
Define las clases de objetos voladores (FlyingShape), mitades cortadas (SplitHalf)
y partículas de efectos (SlashParticle) usando DataClasses.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import pygame

from utils import clamp

@dataclass
class FlyingShape:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: Tuple[int, int, int]
    is_bomb: bool = False

    def update(self, dt: float, gravity: float):
        self.vy += gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, surface: pygame.Surface):
        if self.is_bomb:
            cx, cy = int(self.x), int(self.y)
            pygame.draw.circle(surface, self.color, (cx, cy), self.radius)
            pygame.draw.circle(surface, (220, 220, 220), (cx, cy), self.radius, width=2)
            x_size = int(self.radius * 0.52)
            x_color = (255, 90, 90)
            x_width = max(3, int(self.radius * 0.12))
            pygame.draw.line(surface, x_color, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), width=x_width)
            pygame.draw.line(surface, x_color, (cx - x_size, cy + x_size), (cx + x_size, cy - x_size), width=x_width)
        else:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surface, (250, 250, 250), (int(self.x), int(self.y)), self.radius, width=2)

    @staticmethod
    def create_random(width: int, height: int) -> 'FlyingShape':
        spawn_x = random.randint(int(width * 0.15), int(width * 0.85))
        spawn_y = height + random.randint(20, 80)

        bomb_spawn_chance = 0.16
        if random.random() < bomb_spawn_chance:
            radius = random.randint(38, 54)
            speed_x = random.uniform(-180.0, 180.0)
            speed_y = random.uniform(-980.0, -760.0)
            return FlyingShape(spawn_x, spawn_y, speed_x, speed_y, radius, (55, 55, 60), True)

        shape_profiles = [
            {
                "weight": 0.45,
                "radius_range": (34, 46),
                "vx_range": (-260.0, 260.0),
                "vy_range": (-1060.0, -860.0),
                "colors": [(255, 110, 110), (255, 185, 90)],
            },
            {
                "weight": 0.35,
                "radius_range": (44, 58),
                "vx_range": (-190.0, 190.0),
                "vy_range": (-920.0, -740.0),
                "colors": [(130, 225, 140), (120, 180, 255)],
            },
            {
                "weight": 0.20,
                "radius_range": (58, 72),
                "vx_range": (-130.0, 130.0),
                "vy_range": (-820.0, -650.0),
                "colors": [(220, 140, 255)],
            },
        ]

        profile = random.choices(shape_profiles, weights=[p["weight"] for p in shape_profiles], k=1)[0]
        radius = random.randint(*profile["radius_range"])
        speed_x = random.uniform(*profile["vx_range"])
        speed_y = random.uniform(*profile["vy_range"])
        color = random.choice(profile["colors"])

        return FlyingShape(spawn_x, spawn_y, speed_x, speed_y, radius, color, False)


@dataclass
class SplitHalf:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: Tuple[int, int, int]
    angle: float
    spin: float
    side: int
    life: float

    def update(self, dt: float, gravity: float):
        self.life -= dt
        self.vy += gravity * 0.9 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, surface: pygame.Surface):
        points: List[Tuple[float, float]] = []
        steps = 20
        start = self.angle - math.pi / 2
        for i in range(steps + 1):
            t = i / steps
            a = start + t * math.pi
            px = self.radius * math.cos(a)
            py = self.radius * math.sin(a)
            if self.side < 0:
                px = -px
            
            # Apply rotation based on spin and life
            rotation_angle = self.spin * (1.0 - self.life)
            rx = px * math.cos(rotation_angle) - py * math.sin(rotation_angle)
            ry = px * math.sin(rotation_angle) + py * math.cos(rotation_angle)
            points.append((self.x + rx, self.y + ry))

        points.append((self.x, self.y))
        pygame.draw.polygon(surface, self.color, points)
        pygame.draw.polygon(surface, (245, 245, 245), points, width=2)
    
    @staticmethod
    def create_from_shape(shape: FlyingShape, slash_dx: float, slash_dy: float) -> List['SplitHalf']:
        length = math.hypot(slash_dx, slash_dy)
        if length < 1e-6:
            slash_dx, slash_dy = 1.0, 0.0
            length = 1.0

        nx = -slash_dy / length
        ny = slash_dx / length
        separation_speed = 220.0

        halves = []
        for side in (-1, 1):
            halves.append(
                SplitHalf(
                    x=shape.x + nx * side * 8.0,
                    y=shape.y + ny * side * 8.0,
                    vx=shape.vx + nx * side * separation_speed,
                    vy=shape.vy + ny * side * separation_speed - 80.0,
                    radius=shape.radius,
                    color=shape.color,
                    angle=math.atan2(slash_dy, slash_dx),
                    spin=side * random.uniform(2.5, 4.5),
                    side=side,
                    life=0.9,
                )
            )
        return halves


@dataclass
class SlashParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: Tuple[int, int, int]

    def update(self, dt: float, gravity: float):
        self.life -= dt
        self.vy += gravity * 0.45 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, surface: pygame.Surface):
        alpha_life = clamp(self.life / 0.45, 0.15, 1.0)
        radius = int(2 + alpha_life * 4)
        c_r = int(self.color[0] * alpha_life)
        c_g = int(self.color[1] * alpha_life)
        c_b = int(self.color[2] * alpha_life)
        pygame.draw.circle(surface, (c_r, c_g, c_b), (int(self.x), int(self.y)), radius)

    @staticmethod
    def create_burst(x: float, y: float, color: Tuple[int, int, int]) -> List['SlashParticle']:
        particles = []
        for _ in range(16):
            angle = random.uniform(0.0, math.tau)
            speed = random.uniform(120.0, 420.0)
            particles.append(
                SlashParticle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=random.uniform(0.22, 0.45),
                    color=color,
                )
            )
        return particles
