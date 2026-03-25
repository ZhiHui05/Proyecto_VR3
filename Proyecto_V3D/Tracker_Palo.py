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


def run_tracker(camera_id=1):
    # --- CARGA DE CALIBRACIÓN DE CÁMARA ---
    calibration_file = "camera_calibration.npz"
    mtx, dist = None, None
    camera_matrix_init = False
    
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
    print(f"Envíando datos a {udp_ip_send}:{udp_port_send}")
    print(f"Escuchando estado del juego en puerto 5006")
    print("Presiona q para salir.")

    game_objects = []
    game_halves = []
    last_game_update = 0

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
                game_width = state.get("width", 1280)
                game_height = state.get("height", 720)
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

        # --- Optimización: Mapa de distorsión (Versión Simple por petición del usuario) ---
        if mtx is not None and dist is not None:
             h_cam, w_cam = frame.shape[:2]
             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_cam,h_cam), 0, (w_cam,h_cam))
             frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # --- APLICAR ZOOM DIGITAL (10%) para evitar esquinas muertas ---
        fh, fw = frame.shape[:2]
        crop_ratio = 0.10
        cy_start, cy_end = int(fh * crop_ratio), int(fh * (1 - crop_ratio))
        cx_start, cx_end = int(fw * crop_ratio), int(fw * (1 - crop_ratio))
        
        frame = frame[cy_start:cy_end, cx_start:cx_end]
        frame = cv2.resize(frame, (fw, fh))
        h_cam, w_cam = frame.shape[:2] # Recalcular dimensiones tras resize
        
        # --- 3. Procesamiento (HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rango AZUL/CYAN estándar
        lower_color = np.array([90, 100, 100])
        upper_color = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Momentos (Optimizado: solo buscar si hay píxeles)
        count = cv2.countNonZero(mask)
        cx, cy = 0, 0
        
        if count > 50: # Umbral de ruido
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                # Usar coordenadas reales y enviarlas directamente (sin easing)
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # Visualización Cursor (ROJO y REAL, como solicitado)
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 3) 
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                # Enviar UDP inmediatamente
                data = f"{cx},{cy},{w_cam},{h_cam}"
                try:
                    sock_send.sendto(data.encode(), (udp_ip_send, udp_port_send))
                except:
                    pass

        # --- 4. Renderizado AR (Con blending para feedback visual) ---
        if game_objects or game_halves:
            # Factores de escala
            scale_x = w_cam / game_width
            scale_y = h_cam / game_height
            
            overlay = frame.copy()
            
            for obj in game_objects:
                ox = int(obj["x"] * scale_x)
                oy = int(obj["y"] * scale_y)
                orad = int(obj["r"] * scale_x) 
                ocolor = obj["c"] # RGB
                
                cv_color = (ocolor[2], ocolor[1], ocolor[0]) # BGR
                
                # Renderizar Fruta/Bomba
                cv2.circle(overlay, (ox, oy), orad, cv_color, -1)
                
                if obj["type"] == "bomb":
                    cv2.putText(overlay, "X", (ox-10, oy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            for half in game_halves:
                hx = int(half["x"] * scale_x)
                hy = int(half["y"] * scale_y)
                hrad = int(half["r"] * scale_x)
                hcolor = half["c"]
                if hcolor and len(hcolor) >= 3:
                     cv_color = (hcolor[2], hcolor[1], hcolor[0])
                else:
                     cv_color = (0,0,255)
                
                # Dibujar mitad como semicírculo (simple)
                cv2.ellipse(overlay, (hx, hy), (hrad, hrad), 0, 0, 180, cv_color, -1)
            
            # Aplicar transparencia para que se vea la cámara detrás
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.imshow("Tracker Palo (Activo)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker()
