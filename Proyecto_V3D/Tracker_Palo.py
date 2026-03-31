import cv2
import numpy as np
import socket
import os
import json
import time

# --- CONFIGURACIÓN DE RED ---
# Socket para ENVIAR coordenadas al juego (Puerto 5005)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip_send = "127.0.0.1"
udp_port_send = 5005

# Socket para RECIBIR estado del juego (Puerto 5006)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock_recv.bind(("127.0.0.1", 5006))
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65535) # Aumentar buffer
    sock_recv.setblocking(False) # No bloquear
except OSError:
    print("Puerto 5006 ya en uso. Sincronización visual desactivada.")

# --- VARIABLES GLOBALES PARA CALIBRACIÓN MANUAL (HOMOGRAFÍA) ---
calibration_points = []
homography_matrix = None
game_width_ref = 1280
game_height_ref = 720

def mouse_callback(event, x, y, flags, param):
    global calibration_points, homography_matrix
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Punto {len(calibration_points)} registrado: {x},{y}")
            
            if len(calibration_points) == 4:
                print("Calculando Homografía...")
                # Puntos origen (Cámara)
                src = np.float32(calibration_points)
                # Puntos destino (Juego/Pantalla)
                dst = np.float32([
                    [0, 0],
                    [game_width_ref, 0],
                    [game_width_ref, game_height_ref],
                    [0, game_height_ref]
                ])
                homography_matrix = cv2.getPerspectiveTransform(src, dst)
                print("¡Homografía lista! Ahora el cursor sigue el plano definido.")

def run_tracker(camera_id=0):
    global calibration_points, homography_matrix
    
    # --- CARGA DE CALIBRACIÓN DE CÁMARA (Intrínseca) ---
    calibration_file = "camera_calibration.npz"
    mtx, dist = None, None
    
    possible_paths = [calibration_file, os.path.join("..", calibration_file), os.path.join(os.path.dirname(__file__), calibration_file)]
    found_file = None
    for path in possible_paths:
        if os.path.exists(path):
            found_file = path
            break

    if found_file:
        try:
            with np.load(found_file) as data:
                mtx = data["mtx"]
                dist = data["dist"]
            print(f"Calibración cargada exitosamente.")
        except Exception as e:
            print(f"Error al cargar la calibración: {e}")
    else:
        print(f"AVISO: No se encontró {calibration_file} - Se usará sin corrección.")

    # --- INICIALIZACIÓN DE LA CÁMARA ---
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_id}")
        return

    # Intentar configurar FPS altos
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("Iniciando Tracker Palo...")
    print("--- INSTRUCCIONES ---")
    print("1. Haz CLICK en las 4 esquinas de tu área de juego (Mesa) en sentido horario:")
    print("   Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("2. Una vez definidos los 4 puntos, el cursor se mapeará correctamente.")
    print("3. Presiona R para reiniciar la calibración.")
    print("4. Presiona Q para salir.")

    cv2.namedWindow("Tracker Palo (Activo)")
    cv2.setMouseCallback("Tracker Palo (Activo)", mouse_callback)

    game_objects = []
    game_halves = []
    game_particles = []
    game_score = 0
    game_lives = 3
    game_combo = 0
    is_game_over = False
    last_game_update = 0

    # Inicializar referencias locales con valores por defecto
    local_game_width = game_width_ref
    local_game_height = game_height_ref

    while True:
        # --- 1. Sincronización con el Juego (Drenar buffer) ---
        try:
            current_time = time.time()
            data_recv = None
            
            # Leer TODOS los paquetes pendientes y quedarse con el último
            while True:
                try:
                    packet, _ = sock_recv.recvfrom(65535)
                    data_recv = packet
                except BlockingIOError:
                    break
            
            if data_recv:
                state = json.loads(data_recv.decode())
                game_objects = state.get("entities", [])
                game_halves = state.get("halves", [])
                game_particles = state.get("particles", [])
                game_score = state.get("score", 0)
                game_lives = state.get("lives", 3)
                game_combo = state.get("combo_count", 0)
                is_game_over = state.get("game_over", False)
                # Actualizar referencia local si el juego cambia de tamaño
                local_game_width = state.get("width", 1280)
                local_game_height = state.get("height", 720)
                last_game_update = current_time
        except Exception:
            pass

        # Limpiar objetos si el juego no responde (> 0.5s)
        if time.time() - last_game_update > 0.5:
            game_objects = []
            game_halves = []
            game_particles = []

        # --- 2. Captura ---
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame_raw, 1)
        h_cam, w_cam = frame.shape[:2]

        # --- Optimización: Mapa de distorsión (Versión Simple) ---
        if mtx is not None and dist is not None:
             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_cam,h_cam), 0, (w_cam,h_cam))
             frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # --- 3. Procesamiento (HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rango AZUL/CYAN estándar
        lower_color = np.array([90, 100, 100])
        upper_color = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Momentos
        count = cv2.countNonZero(mask)
        cx, cy = 0, 0
        
        if count > 50: # Umbral de ruido
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # --- APLICAR HOMOGRAFÍA SI ESTÁ LISTA ---
                msg_x, msg_y = cx, cy

                if homography_matrix is not None:
                    # Transformar punto usando la matriz
                    pt_original = np.array([[[cx, cy]]], dtype=np.float32)
                    pt_transformed = cv2.perspectiveTransform(pt_original, homography_matrix)
                    tx = pt_transformed[0][0][0]
                    ty = pt_transformed[0][0][1]
                    
                    msg_x, msg_y = int(tx), int(ty)
                    
                    # Visualizar punto re-proyectado (solo debug)
                    # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
                
                # Visualización Cursor (Verde = Tracking Raw)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)

                # Enviar UDP (Usamos las coordenadas transformadas si existen, o las raw escaladas luego)
                # NOTA: Si usamos homografía, enviamos coordenadas de JUEGO (0-1280), 
                # por lo que "w_cam" en el mensaje debe ser el ancho del juego para que el input_handler no escale doble.
                
                if homography_matrix is not None:
                     data = f"{msg_x},{msg_y},{game_width_ref},{game_height_ref}"
                else:
                     data = f"{cx},{cy},{w_cam},{h_cam}"

                try:
                    sock_send.sendto(data.encode(), (udp_ip_send, udp_port_send))
                except:
                    pass

        # --- DIBUJAR PUNTOS DE CALIBRACIÓN ---
        for i, pt in enumerate(calibration_points):
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(frame, calibration_points[i-1], pt, (0, 0, 255), 2)
        if len(calibration_points) == 4:
            cv2.line(frame, calibration_points[3], calibration_points[0], (0, 0, 255), 2)


        # --- 4. Renderizado AR (Proyectar el juego en la zona delimitada) ---  
        if (game_objects or game_halves or game_particles):

            H_inv = None
            if homography_matrix is not None:
                H_inv = np.linalg.inv(homography_matrix)

            import math
            def euler_to_matrix(rx, ry, rz):
                rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
                cx, sx = math.cos(rx), math.sin(rx)
                cy, sy = math.cos(ry), math.sin(ry)
                cz, sz = math.cos(rz), math.sin(rz)
                
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                return Rz @ Ry @ Rx

            def get_3d_faces(obj):
                sides = obj.get("sides", 0)
                # Escala proporcional a la camara
                nr = obj.get("nr", 0.05)
                # Multiplicamos por la altura del juego para obtener tamaño base, y un factor visual
                r = nr * game_height_ref * 0.9 
                
                verts = []
                faces = []
                
                if obj.get("type") == "bomb":
                    # Icosahedron-based approximation of Dodecahedron for rendering
                    t = (1.0 + math.sqrt(5.0)) / 2.0
                    ico = [
                        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
                    ]
                    ico = [np.array(v)/np.linalg.norm(v) for v in ico]
                    faces_ico = [
                        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
                    ]
                    d_verts = []
                    for a, b, c in faces_ico:
                        cent = (ico[a] + ico[b] + ico[c]) / 3.0
                        d_verts.append((cent / np.linalg.norm(cent)) * r)
                    verts = d_verts
                    faces = []
                    for vi, v in enumerate(ico):
                        face_ids = [fi for fi, f in enumerate(faces_ico) if vi in f]
                        if len(face_ids) == 5:
                            ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                            u = np.cross(v, ref)
                            u = u / np.linalg.norm(u)
                            w = np.cross(v, u)
                            w = w / np.linalg.norm(w)
                            ordered = []
                            for fid in face_ids:
                                p = d_verts[fid]
                                angle = math.atan2(np.dot(p, w), np.dot(p, u))
                                ordered.append((angle, fid))
                            ordered.sort(key=lambda x: x[0])
                            faces.append([fid for _, fid in ordered])
                elif sides == 4: # Cube
                    rc = r * 0.7
                    verts = [
                        (-rc, -rc, -rc), (rc, -rc, -rc), (rc, rc, -rc), (-rc, rc, -rc),
                        (-rc, -rc, rc),  (rc, -rc, rc),  (rc, rc, rc),  (-rc, rc, rc)
                    ]
                    faces = [
                        [0,1,2,3], [5,4,7,6], [4,0,3,7],
                        [1,5,6,2], [4,5,1,0], [3,2,6,7]
                    ]
                elif sides > 2: # Prism and Halves
                    sides = max(3, sides)
                    depth = r
                    base = []
                    top = []
                    for i in range(sides):
                        ang = math.radians((360/sides) * i)
                        x = r * math.cos(ang)
                        y = r * math.sin(ang)
                        verts.append((x, y, -depth/2))
                        verts.append((x, y, depth/2))
                        base.append(i*2)
                        top.append(i*2+1)
                    faces.append(base)
                    faces.append(top)
                    for i in range(sides):
                        ni = (i+1)%sides
                        faces.append([i*2, ni*2, ni*2+1, i*2+1])
                else: 
                    # generic sphere/ball default
                    verts = [
                        (0, r, 0), (r, 0, 0), (0, 0, r),
                        (-r, 0, 0), (0, 0, -r), (0, -r, 0)
                    ]
                    faces = [
                        [0,1,2], [0,2,3], [0,3,4], [0,4,1],
                        [5,2,1], [5,3,2], [5,4,3], [5,1,4]
                    ]
                return np.array(verts, dtype=np.float32), faces

            all_objs = game_objects + game_halves + game_particles
            for obj in all_objs:
                
                if "nx" in obj and "ny" in obj:
                    gx = obj["nx"] * game_width_ref
                    gy = obj["ny"] * game_height_ref
                else: 
                    gx = obj.get("x",0) * (game_width_ref / max(1, local_game_width))
                    gy = obj.get("y",0) * (game_height_ref / max(1, local_game_height))

                c = obj.get("c", [0, 0, 255])
                if len(c) >= 3:
                     base_color = (c[2], c[1], c[0])
                else:
                     base_color = (0,0,255)
                     
                verts, faces = get_3d_faces(obj)
                
                rot_x = obj.get("rot_x", 0)
                rot_y = obj.get("rot_y", 0)
                rot_z = obj.get("rot_z", obj.get("rot", 0))
                
                R = euler_to_matrix(rot_x, -rot_y, -rot_z)
                verts = (R @ verts.T).T
                
                verts[:, 0] += gx
                verts[:, 1] += gy
                # Add scale for z bounce
                z_offset = obj.get("nz", 0.0) * game_width_ref * 0.5
                verts[:, 2] += z_offset
                
                if H_inv is not None:
                    # Apply fake perspective before homography
                    for i in range(len(verts)):
                        verts[i, 0] += verts[i, 2] * 0.2
                        verts[i, 1] -= verts[i, 2] * 0.2
                        
                    pts_g = verts[:, :2].reshape(-1, 1, 2)
                    imgpts = cv2.perspectiveTransform(pts_g, H_inv).reshape(-1, 2)
                else:
                    imgpts = np.zeros((verts.shape[0], 2), dtype=np.int32)
                    for i in range(len(verts)):
                        # fake perspective without homography
                        vx = verts[i, 0] + verts[i, 2] * 0.2
                        vy = verts[i, 1] - verts[i, 2] * 0.2
                        imgpts[i] = [int(vx * (w_cam/game_width_ref)), int(vy * (h_cam/game_height_ref))]
                        
                # Depth sorting locally
                face_params = []
                for face in faces:
                    cz = np.mean([verts[v_idx][2] for v_idx in face])
                    face_params.append((cz, face))
                    
                # draw from back to front
                face_params.sort(key=lambda x: x[0], reverse=True)
                
                for i, (_, face) in enumerate(face_params):
                    poly = np.array([imgpts[v_idx] for v_idx in face], dtype=np.int32)
                    
                    shade = base_color
                    # rudimentary shading
                    if i < len(face_params) - 2:
                        shade = tuple(max(0, int(cv*0.7)) for cv in base_color)
                        
                    cv2.fillPoly(frame, [poly], shade)
                    
                    if obj.get("type") == "bomb":
                        color_line = (0, 0, 255) # Red lines for bomb
                        linewidth = 2
                    elif obj.get("type") == "particle":
                        color_line = shade
                        linewidth = 1
                    else:
                        color_line = (255, 255, 255) if "type" not in obj else (0, 0, 0)
                        linewidth = 1
                        
                    cv2.polylines(frame, [poly], True, color_line, linewidth)

        # --- DIBUJAR INTERFAZ (UI) ---
        # Score
        cv2.putText(frame, f"Score: {game_score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Combo
        if game_combo >= 2:
            cv2.putText(frame, f"{game_combo} COMBO!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
            
        # Vidas (3 Cruces o Círculos)
        for i in range(3):
            x_pos = w_cam - 50 - i * 40
            y_pos = 40
            if i < (3 - game_lives):
                # Cruz roja (vida perdida)
                cv2.line(frame, (x_pos-10, y_pos-10), (x_pos+10, y_pos+10), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(frame, (x_pos+10, y_pos-10), (x_pos-10, y_pos+10), (0, 0, 255), 3, cv2.LINE_AA)
            else:
                # Círculo verde (vida activa)
                cv2.circle(frame, (x_pos, y_pos), 10, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (x_pos, y_pos), 10, (0, 150, 0), 2, cv2.LINE_AA)

        # Game Over Text
        if is_game_over:
            text = "GAME OVER"
            font_scale = 3
            thickness = 6
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cx_t = (w_cam - text_size[0]) // 2
            cy_t = (h_cam + text_size[1]) // 2
            # Borde negro y texto rojo para resaltar
            cv2.putText(frame, text, (cx_t, cy_t), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+4, cv2.LINE_AA)
            cv2.putText(frame, text, (cx_t, cy_t), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow("Tracker Palo (Activo)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            calibration_points = []
            homography_matrix = None
            print("Calibración reiniciada.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker()
