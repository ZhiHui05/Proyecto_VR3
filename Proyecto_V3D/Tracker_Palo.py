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

def run_tracker(camera_id=1):
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
                     data = f"{msg_x},{msg_y},{local_game_width},{local_game_height}"
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
        if (game_objects or game_halves):
            
            # Si hay homografía, necesitamos la matriz inversa para proyectar el juego (plano) -> cámara (distorsionada)
            H_inv = None
            if homography_matrix is not None:
                H_inv = np.linalg.inv(homography_matrix)

            # Función helper para proyectar puntos de juego -> cámara
            def game_to_cam(gx, gy):
                if H_inv is not None:
                    pt_g = np.array([[[gx, gy]]], dtype=np.float32)
                    pt_c = cv2.perspectiveTransform(pt_g, H_inv)
                    return int(pt_c[0][0][0]), int(pt_c[0][0][1])
                else:
                    # Fallback escalar simple
                    return int(gx * (w_cam/local_game_width)), int(gy * (h_cam/local_game_height))

            # Dibujar Objetos
            all_objs = game_objects + game_halves
            for obj in all_objs:
                # Coordenadas centro
                ox, oy = game_to_cam(obj["x"], obj["y"])
                
                # Radio (aproximado, usando un punto en el borde)
                rx, ry = game_to_cam(obj["x"] + obj["r"], obj["y"])
                orad = int(np.hypot(rx - ox, ry - oy))
                
                # Color
                c = obj["c"]
                if len(c) >= 3:
                     cv_color = (c[2], c[1], c[0])
                else:
                     cv_color = (0,0,255)

                # Render
                if "type" in obj: # Es Fruta/Bomba
                    cv2.circle(frame, (ox, oy), orad, cv_color, -1)
                    if obj["type"] == "bomb":
                        cv2.putText(frame, "X", (ox-10, oy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                else: # Es una mitad
                    cv2.ellipse(frame, (ox, oy), (orad, orad), 0, 0, 180, cv_color, -1)

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
