import cv2
import numpy as np
import socket
import os

# --- CONFIGURACIÓN DE RED ---
# Creamos un socket UDP (SOCK_DGRAM) para comunicar las coordenadas al juego
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "127.0.0.1"
udp_port = 5005

def run_tracker(camera_id=1):
    # --- CARGA DE CALIBRACIÓN DE CÁMARA ---
    # Intentamos cargar el archivo de calibración (.npz) generado previamente.
    # Esto es necesario para corregir la distorsión de la lente (efecto ojo de pez).
    #Elegir entre camera_calibration.npz o camera_charuco_calibration.npz dependiendo de cuál se haya generado.
    calibration_file = "camera_calibration.npz"
    mtx, dist = None, None
    
    # Buscamos el archivo en varias rutas posibles
    possible_paths = [calibration_file, os.path.join("..", calibration_file), os.path.join(os.path.dirname(__file__), calibration_file), os.path.join(os.path.dirname(__file__), "..", calibration_file)]
    
    found_file = None
    for path in possible_paths:
        if os.path.exists(path):
            found_file = path
            break

    if found_file:
        try:
            with np.load(found_file) as data:
                mtx = data["mtx"]   # Matriz de la cámara
                dist = data["dist"] # Coeficientes de distorsión
            print(f"Calibración cargada exitosamente desde: {found_file}")
        except Exception as e:
            print(f"Error al cargar la calibración: {e}")
    # --- INICIALIZACIÓN DE LA CÁMARA ---
    else:
        print(f"AVISO: No se encontró {calibration_file} - Se usará sin corrección.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_id}")
        return

    print("Iniciando Tracker Palo (Solo Tracking UDP + Calibración)...")
    print(f"Envíando datos a {udp_ip}:{udp_port}")
    print("Presiona q para salir.")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame_raw, 1)
# --- CORRECCIÓN DE DISTORSIÓN ---
        # Si se cargó la calibración, corregimos la imagen
        if mtx is not None and dist is not None:
            h_cam, w_cam = frame.shape[:2]
            # Calculamos la nueva matriz óptima para la cámara
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_cam,h_cam), 0, (w_cam,h_cam))
            # Aplicamos 'undistort' para aplanar la imagen
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # --- PROCESAMIENTO DE IMAGEN (TRACKING) ---
        # Convertimos a HSV para facilitar la detección de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definimos el rango de color a rastrear (ajustar si cambia la luz)
        lower_color = np.array([90, 100, 100])
        upper_color = np.array([130, 255, 255])
        
        # Creamos una máscara con los píxeles que están en ese rango
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Calculamos los momentos de la imagen binaria para encontrar el centro
        moments = cv2.moments(mask)
        cx, cy = 0, 0
        if moments["m00"] != 0: # Si hay área detectada
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # --- ENVÍO DE DATOS UDP ---
            height, width = frame.shape[:2]
            # Empaquetamos: posición X, posición Y, ancho y alto de referencia
            data = f"{cx},{cy},{width},{height}"
            try:
                sock.sendto(data.encode(), (udp_ip, udp_port))
            except:
                pass # Ignoramos errores de red puntuales
            
            # Dibujamos un círculo y texto en la posición detectada"{cx},{cy},{width},{height}"
            try:
                sock.sendto(data.encode(), (udp_ip, udp_port))
            except:
                pass
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Pos: {cx},{cy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tracker Palo (Calibrado)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker()
