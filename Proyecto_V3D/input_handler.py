"""
Módulo de entrada (InputHandler).
Gestiona la comunicación por sockets UDP para recibir coordenadas del tracker
y las mapea a coordenadas de pantalla del juego.
"""
import socket
from typing import Tuple, Optional
from utils import clamp

class InputHandler:
    def __init__(self, port: int = 5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)

    def close(self):
        self.sock.close()

    def get_pointer_position(self, current_x: int, current_y: int, width: int, height: int) -> Tuple[int, int]:
        px, py = current_x, current_y
        while True:
            try:
                packet, _ = self.sock.recvfrom(128)
            except BlockingIOError:
                break
            
            parsed = self._parse_tracker_message(packet.decode("utf-8", errors="ignore"))
            if parsed is None:
                continue
            
            tx, ty, source_w, source_h = parsed
            px, py = self._map_to_screen(tx, ty, source_w, source_h, width, height)
            
        return px, py

    def _parse_tracker_message(self, message: str) -> Optional[Tuple[float, float, float, float]]:
        parts = [p.strip() for p in message.split(",")]
        if len(parts) not in (2, 4):
            return None

        try:
            x = float(parts[0])
            y = float(parts[1])
            if len(parts) == 4:
                source_w = float(parts[2])
                source_h = float(parts[3])
            else:
                source_w = 640.0
                source_h = 480.0
        except ValueError:
            return None

        if source_w <= 0 or source_h <= 0:
            return None
        return x, y, source_w, source_h

    def _map_to_screen(self, x: float, y: float, source_w: float, source_h: float, target_w: int, target_h: int) -> Tuple[int, int]:
        normalized_x = clamp(x / source_w, 0.0, 1.0)
        normalized_y = clamp(y / source_h, 0.0, 1.0)
        px = int(normalized_x * target_w)
        py = int(normalized_y * target_h)
        return int(clamp(px, 0, target_w - 1)), int(clamp(py, 0, target_h - 1))
