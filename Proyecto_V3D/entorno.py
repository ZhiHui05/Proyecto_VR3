"""
Punto de entrada principal (Main).
Se encarga de parsear los argumentos de la consola e iniciar el bucle del juego.
"""
import argparse
from game import Game

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Demo pygame: corte de figuras con la trayectoria del raton.')
    parser.add_argument('--width', type=int, default=1280, help='Ancho de la ventana.')
    parser.add_argument('--height', type=int, default=720, help='Alto de la ventana.')
    parser.add_argument('--trail-length', type=int, default=36, help='Cantidad de puntos visibles en la traza.')
    args = parser.parse_args()

    if args.width <= 100 or args.height <= 100:
        raise ValueError('--width y --height deben ser mayores a 100')
    if args.trail_length < 6:
        raise ValueError('--trail-length debe ser al menos 6')
    return args

if __name__ == '__main__':
    arguments = parse_args()
    game = Game(arguments.width, arguments.height, arguments.trail_length)
    game.run()
