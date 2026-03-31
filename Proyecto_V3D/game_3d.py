import sys
import socket
import json
import argparse
import random
import math
from typing import List, Tuple
from ursina import *
from random import uniform

try:
    from ursina.shaders import lit_with_shadows_shader as DEFAULT_LIT_SHADER
except Exception:
    DEFAULT_LIT_SHADER = None

# Añadimos el path para importar los módulos locales de Pygame si es necesario, 
# pero aquí usaremos el InputHandler que ya tenías.
try:
    from input_handler import InputHandler
except ImportError:
    print("Asegúrate de ejecutar este script desde el mismo directorio que input_handler.py")
    sys.exit(1)


CUSTOM_MODELS = {}


def _v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_norm(a):
    m = math.sqrt(max(1e-9, _v_dot(a, a)))
    return (a[0] / m, a[1] / m, a[2] / m)


def _build_flat_shaded_mesh(vertices, triangles):
    flat_vertices = []
    flat_normals = []
    flat_uvs = []
    flat_triangles = []

    for tri in triangles:
        ia, ib, ic = tri
        a = vertices[ia]
        b = vertices[ib]
        c = vertices[ic]

        n = _v_norm(_v_cross(_v_sub(b, a), _v_sub(c, a)))
        centroid = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0)

        # Mantener winding consistente hacia afuera para que el relleno y la luz sean estables.
        if _v_dot(n, centroid) < 0:
            b, c = c, b
            n = _v_mul(n, -1.0)

        base = len(flat_vertices)
        flat_vertices.extend([a, b, c])
        flat_normals.extend([n, n, n])
        flat_uvs.extend([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        flat_triangles.append((base, base + 1, base + 2))

    return Mesh(
        vertices=flat_vertices,
        triangles=flat_triangles,
        normals=flat_normals,
        uvs=flat_uvs,
        mode='triangle',
        static=False,
    )


def _make_prism_model(sides=3):
    vertices = []
    for y in (-0.5, 0.5):
        for i in range(sides):
            ang = (2.0 * math.pi * i) / sides
            vertices.append((math.cos(ang) * 0.5, y, math.sin(ang) * 0.5))

    triangles = []
    # Lados
    for i in range(sides):
        ni = (i + 1) % sides
        b0 = i
        b1 = ni
        t0 = i + sides
        t1 = ni + sides
        triangles.append((b0, b1, t1))
        triangles.append((b0, t1, t0))

    # Tapa inferior (abanico)
    for i in range(1, sides - 1):
        triangles.append((0, i + 1, i))

    # Tapa superior (abanico)
    top0 = sides
    for i in range(1, sides - 1):
        triangles.append((top0, top0 + i, top0 + i + 1))

    return _build_flat_shaded_mesh(vertices, triangles)


def _make_pyramid_model(sides=4):
    vertices = [(0.0, 0.6, 0.0)]
    for i in range(sides):
        ang = (2.0 * math.pi * i) / sides
        vertices.append((math.cos(ang) * 0.55, -0.5, math.sin(ang) * 0.55))

    triangles = []
    # Caras laterales
    for i in range(sides):
        ni = 1 + ((i + 1) % sides)
        triangles.append((0, 1 + i, ni))

    # Base
    for i in range(2, sides):
        triangles.append((1, i + 1, i))

    return _build_flat_shaded_mesh(vertices, triangles)


def _make_octahedron_model():
    vertices = [
        (0, 0.7, 0),
        (0.7, 0, 0),
        (0, 0, 0.7),
        (-0.7, 0, 0),
        (0, 0, -0.7),
        (0, -0.7, 0),
    ]
    triangles = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
        (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4),
    ]
    return _build_flat_shaded_mesh(vertices, triangles)


def _make_dodecahedron_model():
    # Construimos el dodecaedro como dual del icosaedro para evitar tablas largas de caras.
    t = (1.0 + math.sqrt(5.0)) / 2.0
    ico_vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    ico_vertices = [_v_norm(v) for v in ico_vertices]

    ico_faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    # Vertices del dodecaedro = centroides de caras del icosaedro.
    d_vertices = []
    for a, b, c in ico_faces:
        centroid = _v_mul(_v_add(_v_add(ico_vertices[a], ico_vertices[b]), ico_vertices[c]), 1.0 / 3.0)
        d_vertices.append(_v_mul(_v_norm(centroid), 0.75))

    # Cada vertice del icosaedro define una cara pentagonal del dodecaedro.
    triangles = []
    for vi, v in enumerate(ico_vertices):
        face_ids = [fi for fi, f in enumerate(ico_faces) if vi in f]
        if len(face_ids) != 5:
            continue

        ref = (1.0, 0.0, 0.0) if abs(v[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = _v_norm(_v_cross(v, ref))
        w = _v_norm(_v_cross(v, u))

        ordered = []
        for fid in face_ids:
            p = d_vertices[fid]
            angle = math.atan2(_v_dot(p, w), _v_dot(p, u))
            ordered.append((angle, fid))
        ordered.sort(key=lambda x: x[0])
        ids = [fid for _, fid in ordered]

        # Triangulamos pentagono en abanico
        triangles.append((ids[0], ids[1], ids[2]))
        triangles.append((ids[0], ids[2], ids[3]))
        triangles.append((ids[0], ids[3], ids[4]))

    return _build_flat_shaded_mesh(d_vertices, triangles)


def get_model(name):
    if name in CUSTOM_MODELS:
        return CUSTOM_MODELS[name]

    if name == 'tri_prism':
        CUSTOM_MODELS[name] = _make_prism_model(3)
    elif name == 'hex_prism':
        CUSTOM_MODELS[name] = _make_prism_model(6)
    elif name == 'cylinder':
        CUSTOM_MODELS[name] = _make_prism_model(16)
    elif name == 'pyramid':
        CUSTOM_MODELS[name] = _make_pyramid_model(4)
    elif name == 'octahedron':
        CUSTOM_MODELS[name] = _make_octahedron_model()
    elif name == 'dodecahedron':
        CUSTOM_MODELS[name] = _make_dodecahedron_model()
    else:
        CUSTOM_MODELS[name] = name

    return CUSTOM_MODELS[name]

class Game3D:
    def __init__(self, use_udp_input=True):
        # Configurar ventana
        window.title = 'Demo Fruit Ninja 3D'
        window.borderless = False
        window.fullscreen = False
        window.exit_button.visible = False
        window.fps_counter.enabled = True
        
        # Ocultar el cursor por defecto del SO
        mouse.visible = False
        
        # Color del cielo/fondo
        window.color = color.rgb(30, 30, 40)
        
        # Cámara principal
        camera.position = (0, 0, -20)
        camera.fov = 60
        
        # Variables de estado del juego
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_over_timer = 0
        
        # Inicializar el handler UDP para recibir coordenadas de Tracker_Palo
        self.input_handler = InputHandler(use_socket=use_udp_input)
        self.pointer_x = int(window.size[0]) // 2
        self.pointer_y = int(window.size[1]) // 2
        
        # UI
        self.score_text = Text(text=f'Score: {self.score}', position=(-0.8, 0.45), scale=2, color=color.white)
        
        # Sistema de Combo
        self.combo_count = 0
        self.combo_timer = 0.0
        self.combo_text = Text(text='', position=(0, 0.35), scale=3.5, color=color.yellow, origin=(0,0), enabled=False)
        
        # Contadores de vidas (3 X vacías/grises que se pondrán rojas)
        self.crosses = []
        for i in range(3):
            # Posición en el centro superior de la pantalla
            t = Text(text='X', position=(-0.08 + i * 0.08, 0.45), origin=(0,0), scale=3.5, color=color.rgba(150, 150, 150, 100))
            self.crosses.append(t)
            
        self.game_over_text = Text(text='GAME OVER', position=(0, 0), scale=4, color=color.red, origin=(0,0), enabled=False)
        self.restart_text = Text(text='Reiniciando...', position=(0, -0.1), scale=1.5, color=color.light_gray, origin=(0,0), enabled=False)

        # Iluminación y entorno
        self.setup_lighting()
        
        # Un panel posterior tipo muro (para atrapar las sombras)
        self.background_wall = Entity(
            model='quad', 
            color=color.dark_gray, # Color gris oscuro
            texture='white_cube',  # Añadir textura base ayuda a que las sombras y luces resalten mejor
            scale=(60, 40), 
            position=(0, 0, 5), 
            rotation_x=0
        )
        
        # Cursor / Espada y estela
        # Usamos un pequeño cubo como cursor en el espacio 3D
        self.cursor = Entity(model='sphere', color=color.rgba(100,255,100,200), scale=0.5, unlit=True)
        # Lista para la estela (trail) del ratón
        self.trail_entities = []
        for _ in range(10):
            self.trail_entities.append(Entity(model='quad', color=color.rgba(200, 200, 255, 150), scale=0.3, unlit=True))
            
        self.trail_positions = []
        
        # Control de spawns por oleadas
        self.time_since_wave = 0.0
        self.next_wave_in = 0.35
        self.pending_spawns = []
        
        # Comunicacion hacia el tracker (Opcional, similar a game.py)
        self.sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tracker_address = ("127.0.0.1", 5006)
        
        # Frutas activas
        self.active_fruits = []

        # Limites del "escenario jugable" (mundo 3D)
        self.launch_y = -11.5
        self.bottom_limit_y = -15.0
        self.top_limit_y = 7.8
        self.min_apex_y = 5.6
        self.max_apex_y = 7.2

    def setup_lighting(self):
        # Luz direccional principal fuerte para resaltar las caras e iluminar volumen
        pivot = Entity()
        self.sun = DirectionalLight(parent=pivot, y=5, z=5, shadows=True)
        self.sun.look_at(Vec3(-2, -3, -5)) # Más ángulo para generar contraste en polígonos
        self.sun.color = color.white

        # Luz de relleno frontal
        self.fill_light = PointLight(parent=camera, y=1.5, z=-8)
        self.fill_light.color = color.rgba(210, 210, 220, 0.45) # Más intenso pero suave
        
        # Ambiente un pelín más oscuro para crear más sombra tridimensional.
        AmbientLight(color=color.rgba(90, 90, 100, 0.5))

    def reset_game(self):
        self.score = 0
        self.lives = 3
        self.combo_count = 0
        self.combo_timer = 0.0
        self.combo_text.enabled = False
        self.update_ui()
        self.game_over = False
        self.game_over_text.enabled = False
        self.restart_text.enabled = False
        
        for f in self.active_fruits:
            destroy(f)
        self.active_fruits.clear()

    def update_ui(self):
        self.score_text.text = f'Score: {self.score}'
        
        # Actualizamos las cruces de vidas
        lost_lives = 3 - max(0, self.lives)
        for i in range(3):
            if i < lost_lives:
                self.crosses[i].color = color.red
            else:
                self.crosses[i].color = color.rgba(150, 150, 150, 100) # Gises / "Vacias"

def update():
    # Esta función global es llamada por Ursina cada frame. 
    # Delegamos al objeto del juego.
    game_instance.update()

class FloatingObject(Entity):
    def __init__(self, is_bomb=False, spawn_x=None, lateral_velocity=None):
        
        # Definir frutas simulando formas 3D deformando esferas base
        # (Si añadieras un modelo, cambia 'model': 'sphere' por 'model': 'mi_fruta.obj')
        if is_bomb:
            obj_color = color.black
            obj_model = 'sphere'
            obj_scale = 1.3
            mass = 1.2
        else:
            # Catalogo de formas geometricas variadas para estilo arcade 3D.
            frutas = [
                {'name': 'Prisma triangular', 'color': color.azure, 'scale': Vec3(1.1, 1.6, 1.1), 'model': get_model('tri_prism'), 'mass': 0.85},
                {'name': 'Prisma hexagonal', 'color': color.lime, 'scale': Vec3(1.1, 1.5, 1.1), 'model': get_model('hex_prism'), 'mass': 0.95},
                {'name': 'Piramide', 'color': color.yellow, 'scale': Vec3(1.2, 1.8, 1.2), 'model': get_model('pyramid'), 'mass': 0.9},
                {'name': 'Cilindro', 'color': color.orange, 'scale': Vec3(1.0, 1.8, 1.0), 'model': get_model('cylinder'), 'mass': 1.0},
                {'name': 'Octaedro', 'color': color.red, 'scale': Vec3(1.4, 1.4, 1.4), 'model': get_model('octahedron'), 'mass': 0.8},
                {'name': 'Dodecaedro', 'color': color.green, 'scale': Vec3(1.55, 1.55, 1.55), 'model': get_model('dodecahedron'), 'mass': 1.25},
                {'name': 'Poliedro cubico', 'color': color.violet, 'scale': Vec3(1.4, 1.4, 1.4), 'model': 'cube', 'mass': 1.1},
            ]
            f = random.choice(frutas)
            obj_color = f['color']
            obj_model = f['model']
            obj_scale = f['scale']
            mass = f['mass']

        super().__init__(
            model=obj_model,
            color=obj_color,
            texture=None,
            scale=obj_scale,
            position=(spawn_x if spawn_x is not None else uniform(-7.5, 7.5), game_instance.launch_y, 0),
            collider='sphere' if is_bomb else 'box'
        )
        self.is_bomb = is_bomb
        self.uses_custom_mesh = isinstance(self.model, Mesh)
        self.wireframe = False
        self.unlit = False
        
        # Aplicamos el shader de iluminacion y sombras a todos los modelos (Ursina default o custom meshes)
        if DEFAULT_LIT_SHADER is not None:
            self.shader = DEFAULT_LIT_SHADER
            
        self.color = obj_color
        self.double_sided = True
        
        # Fisica tipo Fruit Ninja: arco limpio sin salirse por arriba.
        # Elegimos altura objetivo y derivamos v0 con v^2 = 2 * g * h.
        self.gravity = -uniform(10.0, 12.0) * (0.9 + 0.2 * mass)
        half_h = max(0.2, self.scale.y * 0.5)
        target_apex = uniform(game_instance.min_apex_y, game_instance.max_apex_y) - half_h
        delta_h = max(1.0, target_apex - self.y)
        jump_velocity = (2.0 * abs(self.gravity) * delta_h) ** 0.5

        self.velocity = Vec3(lateral_velocity if lateral_velocity is not None else uniform(-1.8, 1.8), jump_velocity, 0)
        self.rotation_speed = Vec3(uniform(-200, 200), uniform(-200, 200), uniform(-200, 200))

        if self.is_bomb:
            # Marca visual tipo Fruit Ninja: X roja sobre la bomba.
            x_thickness = 0.12
            x_length = 1.3
            x_depth = 0.08
            Entity(
                parent=self,
                model='cube',
                color=color.red,
                texture=None,
                scale=(x_thickness, x_length, x_depth),
                position=(0, 0, 0.72),
                rotation_z=45,
                unlit=False,
                shader=DEFAULT_LIT_SHADER,
                double_sided=True,
                collider=None,
            )
            Entity(
                parent=self,
                model='cube',
                color=color.red,
                texture=None,
                scale=(x_thickness, x_length, x_depth),
                position=(0, 0, 0.72),
                rotation_z=-45,
                unlit=False,
                shader=DEFAULT_LIT_SHADER,
                double_sided=True,
                collider=None,
            )
        
        # Guardamos referencia para el manager
        game_instance.active_fruits.append(self)

    def update(self):
        # Aplicar gravedad y mover
        self.velocity.y += self.gravity * time.dt
        self.position += self.velocity * time.dt
        self.rotation += self.rotation_speed * time.dt

        # Tope superior sin rebote: si llega al limite, corta la subida y cae.
        top_cap = game_instance.top_limit_y - max(0.2, self.scale.y * 0.5)
        if self.y > top_cap and self.velocity.y > 0:
            self.y = top_cap
            self.velocity.y = 0

        # Si cae muy por debajo, se destruye
        if self.y < game_instance.bottom_limit_y:
            if not self.is_bomb and not game_instance.game_over:
                game_instance.lives -= 1
                game_instance.update_ui()
                if game_instance.lives <= 0:
                    game_instance.handle_game_over()
                    
            if self in game_instance.active_fruits:
                game_instance.active_fruits.remove(self)
            destroy(self)

    def slice(self, slice_dir):
        if self in game_instance.active_fruits:
            game_instance.active_fruits.remove(self)

        if self.is_bomb:
            # Efecto explosión
            game_instance.lives -= 1
            game_instance.update_ui()
            
            # Perder combo al golpear bomba
            game_instance.combo_count = 0
            game_instance.combo_text.enabled = False
            
            self.create_particles(color.red, 15, speed=10)
            if game_instance.lives <= 0:
                game_instance.handle_game_over()
        else:
            game_instance.score += 1
            
            # Registrar combo
            game_instance.combo_count += 1
            game_instance.combo_timer = 0.5 # medio segundo para enlazar el siguiente corte
            
            if game_instance.combo_count >= 2:
                game_instance.combo_text.enabled = True
                game_instance.combo_text.text = f'{game_instance.combo_count} COMBO!'
                # Pequeño bonus de puntos por combo continuo
                if game_instance.combo_count >= 3:
                    game_instance.score += 1
            
            game_instance.update_ui()
            # Crear las dos mitades
            self.create_halves(slice_dir)
            self.create_particles(self.color, 5, speed=5)
            
        destroy(self)

    def create_halves(self, slice_dir):
        # Escalar las mitades para que visualmente parezca que se metió un tajo por en medio (más finas)
        half_scale = Vec3(self.scale.x * 0.5, self.scale.y * 0.9, self.scale.z * 0.9)
        half_shader = DEFAULT_LIT_SHADER

        half1 = Entity(model=self.model, color=self.color, texture=None, scale=half_scale, position=self.position, shader=half_shader, unlit=False, double_sided=True)
        half2 = Entity(model=self.model, color=self.color, texture=None, scale=half_scale, position=self.position, shader=half_shader, unlit=False, double_sided=True)
        
        # Las desplazamos un pelín antes de enviarlas a volar para que no se solapen
        dir_vector = Vec3(slice_dir.y, -slice_dir.x, 0).normalized() 
        half1.position += dir_vector * 0.3
        half2.position -= dir_vector * 0.3
        
        # Efecto de explosión violenta hacia los lados al cortarse (como en Fruit Ninja)
        throw_force = 12
        half1.animate_position(half1.position + dir_vector * throw_force + Vec3(0, -15, 0), duration=1.5, curve=curve.linear)
        half2.animate_position(half2.position - dir_vector * throw_force + Vec3(0, -15, 0), duration=1.5, curve=curve.linear)
        
        # Rotación súper loca al cortarse para vender el efecto
        half1.animate_rotation(Vec3(uniform(300, 600), uniform(300, 600), uniform(300, 600)), duration=1.1)
        half2.animate_rotation(Vec3(uniform(-600, -300), uniform(-300, -600), uniform(-600, -300)), duration=1.1)
        
        destroy(half1, delay=1.5)
        destroy(half2, delay=1.5)

    def create_particles(self, p_color, count, speed):
        for _ in range(count):
            p = Entity(model='cube', color=p_color, texture=None, scale=0.3, position=self.position, unlit=True, double_sided=True)
            direction = Vec3(uniform(-1,1), uniform(-1,1), uniform(-1,1)).normalized()
            p.animate_position(p.position + direction * speed, duration=0.8, curve=curve.out_expo)
            p.animate_scale(0, duration=0.8)
            destroy(p, delay=0.8)

# Metodos para el bucle principal de Game3D
def update_game_instance(self):
    game = self
    
    if game.game_over:
        game.game_over_timer -= time.dt
        if game.game_over_timer <= 0:
            game.reset_game()
        return
        
    # Decadencia del timer de combo
    if game.combo_timer > 0:
        game.combo_timer -= time.dt
        if game.combo_timer <= 0:
            if game.combo_count >= 3:
                # Mostrar un extra por combo (opcional) o simplemente reset
                pass
            game.combo_count = 0
            game.combo_text.enabled = False

    # 1. ACTUALIZAR CURSOR DESDE UDP (O RATON)
    # Por defecto ursina captura el ratón. Usamos input_handler para leer desde Tracker si manda UDP
    game.pointer_x, game.pointer_y = game.input_handler.get_pointer_position(
        game.pointer_x, game.pointer_y, int(window.size[0]), int(window.size[1])
    )
    
    # 2. MAPEAR A COORDENADAS 3D
    # Ursina tiene un sistema de pantalla que va de -0.5 a 0.5 en UI, pero el mundo real depende de fov de camara.
    # Convertimos coordenadas de píxeles (0-Width, 0-Height) a unidades de mundo (plano z=0).
    # La coordenada 'pointer_x' va de 0 a WindowWidth.
    aspect_ratio = window.aspect_ratio
    # Calculamos pos en plano
    world_width = 20 * aspect_ratio 
    world_height = 20 
    
    # Si tenemos mouse de ursina y no viene por UDP, podemos usar mouse.x / mouse.y
    # Si viene por UDP (pointer_x, pointer_y en pixeles):
    cursor_world_x = (game.pointer_x / max(1, int(window.size[0])) - 0.5) * world_width * 1.5
    cursor_world_y = -(game.pointer_y / max(1, int(window.size[1])) - 0.5) * world_height * 1.5
    
    # Cursor position smoothing
    target_pos = Vec3(cursor_world_x, cursor_world_y, 0)
    game.cursor.position = lerp(game.cursor.position, target_pos, min(1.0, time.dt * 30))
    game.cursor.rotation_z += 200 * time.dt
    
    # 3. ACTUALIZAR ESTELA (TRAIL)
    game.trail_positions.append(game.cursor.position)
    if len(game.trail_positions) > len(game.trail_entities):
        game.trail_positions.pop(0)
        
    for i, pos in enumerate(game.trail_positions):
        game.trail_entities[i].position = pos
        # Hacer que se encojan hacia el final de la estela
        scale_fac = (i / len(game.trail_entities)) * 0.4
        game.trail_entities[i].scale = scale_fac

    # 4. COMPROBAR COLISIONES CON CORTE
    if len(game.trail_positions) >= 2:
        p1 = game.trail_positions[-1]
        p2 = game.trail_positions[-2]
        
        # Solo cortamos si el cursor se movió lo suficiente (evitar cortar dejando el cursor quieto)
        slice_dist = distance(p1, p2)
        if slice_dist > 0.1:
            slice_dir = (p1 - p2).normalized()
            
            # Revisar todas las frutas y ver si chocan con la espada/cursor
            # Una forma sencilla es distancia del cursor a la fruta
            for f in list(game.active_fruits):
                dist_to_fruit = distance(game.cursor.position, f.position)
                # Usar la escala más alta de la fruta al colisionar por si está deformada (e.g. un plátano largo)
                col_radius = max(f.scale_x, f.scale_y)
                if dist_to_fruit < col_radius: # Radio de colisión
                    f.slice(slice_dir)

    # 5. SPAWN DE OBJETOS POR OLEADAS
    game.time_since_wave += time.dt
    if game.time_since_wave >= game.next_wave_in:
        game.time_since_wave = 0.0
        game.next_wave_in = uniform(1.2, 2.0)
        game.schedule_wave()

    if game.pending_spawns:
        next_pending = []
        for delay, is_bomb, spawn_x, lateral_velocity in game.pending_spawns:
            delay -= time.dt
            if delay <= 0:
                FloatingObject(is_bomb=is_bomb, spawn_x=spawn_x, lateral_velocity=lateral_velocity)
            else:
                next_pending.append((delay, is_bomb, spawn_x, lateral_velocity))
        game.pending_spawns = next_pending

    # 6. ENVIAR ESTADO AL TRACKER (Opcional)
    game.send_state()
    
def handle_game_over(game):
    game.game_over = True
    game.game_over_text.enabled = True
    game.restart_text.enabled = True
    game.game_over_timer = 3.0

def schedule_wave(game):
    # Oleadas estilo Fruit Ninja: grupos compactos con un ligero desfase temporal.
    count = random.randint(2, 5)
    center_x = uniform(-4.5, 4.5)
    spacing = uniform(1.2, 1.9)
    spawn_step = uniform(0.07, 0.13)

    # Reducimos la probabilidad de que la oleada incluya una bomba 
    include_bomb = uniform(0, 1) < 0.15 # Alrededor de un 15% de probabilidad por oleada
    bomb_index = random.randint(0, count - 1) if include_bomb else -1

    for i in range(count):
        slot = i - (count - 1) / 2.0
        spawn_x = center_x + slot * spacing
        spawn_x = max(-7.4, min(7.4, spawn_x))

        lateral_velocity = slot * 0.35 + uniform(-0.15, 0.15)
        is_bomb = i == bomb_index
        delay = i * spawn_step
        game.pending_spawns.append((delay, is_bomb, spawn_x, lateral_velocity))

Game3D.update = update_game_instance
Game3D.handle_game_over = handle_game_over
Game3D.schedule_wave = schedule_wave

def send_state(game):
    entities = []
    for f in game.active_fruits:
        entities.append({
            "x": int((f.x / 30 + 0.5) * int(window.size[0])), 
            "y": int((-f.y / 30 + 0.5) * int(window.size[1])),
            "type": "bomb" if f.is_bomb else "fruit"
        })
    state = {
        "width": int(window.size[0]),
        "height": int(window.size[1]),
        "entities": entities,
        "halves": []
    }
    try:
        msg = json.dumps(state)
        game.sender_sock.sendto(msg.encode(), game.tracker_address)
    except:
        pass
        
Game3D.send_state = send_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo Fruit Ninja 3D (Ursina)")
    parser.add_argument("--no-udp", action="store_true", help="Desactivar input remoto UDP (Usar ratón y teclado de SO)")
    args = parser.parse_args()
    
    # Ursina debe instanciarse antes de crear Entidades
    app = Ursina()
    
    # Permitir al usuario visualizar el ratón de SO si no se usa UDP para testeo
    mouse.visible = args.no_udp

    game_instance = Game3D(use_udp_input=not args.no_udp)
    
    app.run()
