import cv2
import numpy as np
import socket
import os

# Configuración del Socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "127.0.0.1"
udp_port = 5005

# -----------------------------------------------------------
# CARGAR CALIBRACIÓN DE CÁMARA (Para Realidad Aumentada)
# -----------------------------------------------------------
# Busca el archivo 'camera_calibration.npz' generado por calibracion.py.
# Si existe, se usará para corregir la distorsión de la lente ("undistort").
calibration_file = "camera_calibration.npz"
mtx, dist = None, None

if os.path.exists(calibration_file):
    try:
        with np.load(calibration_file) as data:
            mtx = data['mtx']
            dist = data['dist']
        print(f"Calibración cargada exitosamente desde: {calibration_file}")
    except Exception as e:
        print(f"Error al cargar la calibración: {e}")
else:
    print("AVISO: No se encontró 'camera_calibration.npz'. La imagen NO tendrá corrección de lente.")
# -----------------------------------------------------------

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------------------------
    # APLICAR CORRECCIÓN DE LENTE (Undistort)
    # -----------------------------------------------------------
    if mtx is not None and dist is not None:
        h, w = frame.shape[:2]
        # Obtener nueva matriz óptima para evitar bordes negros indeseados
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # Corregir la imagen
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # Opcional: Recortar la imagen a la región de interés válida
        # x, y, w, h = roi
        # frame = frame[y:y+h, x:x+w]
    # -----------------------------------------------------------

    # Voltear horizontalmente para que actúe como espejo
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detectar color (ejemplo palo rojo)
    # Ajusta estos valores según el color de tu palo
    #lower_red = np.array([0, 120, 70])
    #upper_red = np.array([10, 255, 255])
    
    # Máscara para el color rojo (rango bajo)
    #mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Máscara para el color rojo (rango alto, el rojo da la vuelta en HSV)
    #lower_red2 = np.array([170, 120, 70])
    #upper_red2 = np.array([180, 255, 255])
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # RANGO AZUL
    # H: 90-130 | S: 100-255 | V: 100-255
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calcular momentos para encontrar el centro
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Dibujar un círculo en el centro detectado
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        
        # Enviar coordenadas al entorno (formato "x,y,w,h")
        frame_h, frame_w = frame.shape[:2]
        data = f"{cx},{cy},{frame_w},{frame_h}"
        sock.sendto(data.encode(), (udp_ip, udp_port))
        print(f"Enviando: {data}")

    cv2.imshow("Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27: # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()