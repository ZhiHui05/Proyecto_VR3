"""
Utilidades matemáticas y de geometría (Utils).
Contiene funciones auxiliares para detección de colisiones, distancias
entre puntos y cálculo de intersecciones.
"""
import math

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))

def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-8:
        return math.dist((px, py), (ax, ay))
    t = clamp((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.dist((px, py), (cx, cy))

def segment_intersects_circle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, radius: float) -> bool:
    return point_to_segment_distance(cx, cy, ax, ay, bx, by) <= radius
