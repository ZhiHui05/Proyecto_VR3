"""
Clase principal del juego (Game).
Controla el bucle principal, gestión de eventos, actualizaciones de estado y renderizado.
"""
import pygame
import argparse
import socket
import json
from typing import List, Tuple
from entities import FlyingShape, SplitHalf, SlashParticle
from input_handler import InputHandler
from utils import clamp, segment_intersects_circle

class Game:
    def __init__(self, width: int, height: int, trail_length: int, use_udp_input: bool = True):
        pygame.init()
        pygame.display.set_caption("Demo Fruit Ninja (OOP Split) - control con raton")
        
        self.width = width
        self.height = height
        self.trail_length = trail_length
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.game_over_font = pygame.font.SysFont("consolas", 64, bold=True)
        self.game_over_sub_font = pygame.font.SysFont("consolas", 24)
        
        pygame.mouse.set_visible(False)
        
        # En modo AR no necesitamos el socket UDP a menos que queramos input externo adicional
        # Evita conflictos de puerto si ejecutamos game.py directamente en modo AR
        self.input_handler = InputHandler(use_socket=use_udp_input)
        
        # Socket para enviar estado al Tracker (Feedback AR)
        self.sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tracker_address = ("127.0.0.1", 5006)
        
        # Game constants
        self.gravity = 980.0
        self.spawn_interval_start = 1.20
        self.spawn_interval_end = 0.25
        self.difficulty_ramp_seconds = 75.0
        self.game_over_duration = 2.0
        
        # State
        self.running = True
        self.game_over = False
        self.game_over_timer = 0.0
        self.pointer_x = width // 2
        self.pointer_y = height // 2
        
        # Estructura para fondo dinámico (AR)
        self.background_surface = None
        self.background_color = (18, 18, 22) # Default dark background
        
        self.reset_game_state()

    def reset_game_state(self):
        self.score = 0
        self.lives = 3
        self.shapes: List[FlyingShape] = []
        self.split_halves: List[SplitHalf] = []
        self.particles: List[SlashParticle] = []
        self.trail: List[Tuple[float, float]] = []
        self.spawn_timer = 0.0
        self.elapsed_time = 0.0

    def run(self):
        while self.running:
            dt = self.clock.tick(144) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
        
        self.input_handler.close()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
        
        # Update pointer
        self.pointer_x, self.pointer_y = self.input_handler.get_pointer_position(
            self.pointer_x, self.pointer_y, self.width, self.height
        )

    def _update(self, dt: float):
        self.elapsed_time += dt

        if self.game_over:
            self.game_over_timer -= dt
            if self.game_over_timer <= 0.0:
                self.reset_game_state()
                self.game_over = False
            return

        # Trail logic
        self.trail.append((float(self.pointer_x), float(self.pointer_y)))
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)

        # Spawning
        difficulty_t = clamp(self.elapsed_time / self.difficulty_ramp_seconds, 0.0, 1.0)
        current_spawn_interval = (
            self.spawn_interval_start + 
            (self.spawn_interval_end - self.spawn_interval_start) * difficulty_t
        )
        
        self.spawn_timer += dt
        while self.spawn_timer >= current_spawn_interval:
            self.shapes.append(FlyingShape.create_random(self.width, self.height))
            self.spawn_timer -= current_spawn_interval

        # Enviar estado del juego al tracker
        self._send_game_state()

        # Update shapes
        for shape in self.shapes:
            shape.update(dt, self.gravity)

        # Collisions
        self._check_collisions(dt)
        
        # Cleanup
        self._cleanup_entities(dt)


    def _send_game_state(self):
        """Envía la lista de objetos activos al tracker vía UDP para visualización remota."""
        entities = []
        for s in self.shapes:
            entities.append({
                "x": int(s.x),
                "y": int(s.y),
                "r": int(s.radius),
                "c": s.color,
                "type": "bomb" if s.is_bomb else "fruit"
            })
        
        # Enviamos también las dimensiones para poder escalar en el otro lado
        halves = []
        for h in self.split_halves:
            halves.append({
                "x": int(h.x),
                "y": int(h.y),
                "r": int(h.radius),
                "c": h.color
            })

        state = {
            "width": self.width,
            "height": self.height,
            "entities": entities,
            "halves": halves
        }
        
        try:
            msg = json.dumps(state)
            self.sender_sock.sendto(msg.encode(), self.tracker_address)
        except Exception:
            pass

    def _check_collisions(self, dt: float):
        to_remove = set()
        if len(self.trail) >= 2:
            ax, ay = self.trail[-2]
            bx, by = self.trail[-1]
            slash_dx = bx - ax
            slash_dy = by - ay
            
            for idx, shape in enumerate(self.shapes):
                if segment_intersects_circle(ax, ay, bx, by, shape.x, shape.y, shape.radius):
                    to_remove.add(idx)
                    if shape.is_bomb:
                        self.lives -= 1
                        self.particles.extend(SlashParticle.create_burst(shape.x, shape.y, (255, 80, 80)))
                        if self.lives <= 0:
                            self.game_over = True
                            self.game_over_timer = self.game_over_duration
                            break # End checking collisions if dead
                    else:
                        self.score += 1
                        self.particles.extend(SlashParticle.create_burst(shape.x, shape.y, shape.color))
                        self.split_halves.extend(SplitHalf.create_from_shape(shape, slash_dx, slash_dy))
        
        if to_remove:
            self.shapes = [s for i, s in enumerate(self.shapes) if i not in to_remove]

    def _cleanup_entities(self, dt: float):
        # Remove shapes out of screen
        active_shapes = []
        for shape in self.shapes:
            if shape.y - shape.radius > self.height + 60:
                if not shape.is_bomb:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                        self.game_over_timer = self.game_over_duration
                # Dropped shape is removed
            elif shape.x < -120 or shape.x > self.width + 120:
                pass # Removed
            else:
                active_shapes.append(shape)
        
        if not self.game_over:
            self.shapes = active_shapes

        # Update and cleanup particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update(dt, self.gravity)

        # Update and cleanup halves
        active_halves = []
        for half in self.split_halves:
            if half.life > 0 and half.y - half.radius <= self.height + 100:
                half.update(dt, self.gravity)
                active_halves.append(half)
        self.split_halves = active_halves

    def _draw(self):
        # Dibujar fondo (AR o Color sólido)
        if self.background_surface is not None:
            # Escalar si es necesario
            if self.background_surface.get_size() != (self.width, self.height):
                bg = pygame.transform.scale(self.background_surface, (self.width, self.height))
                self.screen.blit(bg, (0, 0))
            else:
                self.screen.blit(self.background_surface, (0,0))
        else:
            self.screen.fill(self.background_color)

        # Draw Fruits/Bombs
        for shape in self.shapes:
            shape.draw(self.screen)
        
        # Draw Split Halves
        for half in self.split_halves:
            half.draw(self.screen)
            
        # Draw Particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw Trail
        if len(self.trail) >= 2:
            pygame.draw.lines(self.screen, (200, 200, 255), False, self.trail, 3)

        # Draw Sword Cursor
        pygame.draw.circle(self.screen, (100, 255, 100), (self.pointer_x, self.pointer_y), 10, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (self.pointer_x, self.pointer_y), 3)

        # UI
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))
        
        lives_text = self.font.render(f"Lives: {self.lives}", True, (255, 100, 100))
        self.screen.blit(lives_text, (self.width - 120, 20))
        
        if self.game_over:
            go_text = self.game_over_font.render("GAME OVER", True, (255, 50, 50))
            w, h = go_text.get_size()
            self.screen.blit(go_text, (self.width//2 - w//2, self.height//2 - h//2 - 20))
            
            sub_text = self.game_over_sub_font.render("Reiniciando...", True, (200, 200, 200))
            sw, sh = sub_text.get_size()
            self.screen.blit(sub_text, (self.width//2 - sw//2, self.height//2 + h//2 + 10))

        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Fruit Ninja (OOP Split)")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de pantalla")
    parser.add_argument("--height", type=int, default=720, help="Ancho de pantalla")
    parser.add_argument("--trail", type=int, default=15, help="Longitud de estela")
    parser.add_argument("--no-udp", action="store_true", help="Desactivar input remoto UDP")
    
    args = parser.parse_args()
    
    game = Game(args.width, args.height, args.trail, use_udp_input=not args.no_udp)
    game.run()
