import cv2
import numpy as np
import socket

# Configuración del Socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "127.0.0.1"
udp_port = 5005

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # RANGO VERDE/VERDOSO
    # H: 35-85 | S: 100-255 | V: 100-255
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green, upper_green)

    # RANGO COMBINADO (AMARILLO + VERDE)
    # Capta desde amarillo puro hasta verde intenso (H: 20 a 85)
    lower_yellow_green = np.array([20, 100, 100])
    upper_yellow_green = np.array([85, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green, upper_green)


    mask = mask1 + mask2

    # Calcular momentos para encontrar el centro
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Dibujar un círculo en el centro detectado
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        
        # Enviar coordenadas a Unity (formato "x,y")
        data = f"{cx},{cy}"
        sock.sendto(data.encode(), (udp_ip, udp_port))
        print(f"Enviando: {data}")

    cv2.imshow("Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27: # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()