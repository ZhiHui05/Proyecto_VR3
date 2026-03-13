import cv2
import numpy as np

cap = cv2.VideoCapture(0)

tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RANGOS ROJO (doble rango)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # RANGO AMARILLO (suele ser único rango)
    # H: 20-35 | S: 100-255 | V: 100-255
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # RANGO VERDE/VERDOSO
    # H: 35-85 | S: 100-255 | V: 100-255
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([85, 255, 255])

    # RANGO COMBINADO (AMARILLO + VERDE)
    # Capta desde amarillo puro hasta verde intenso (H: 20 a 85)
    lower_yellow_green = np.array([20, 100, 100])
    upper_yellow_green = np.array([85, 255, 255])

    # Limpiar ruido
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------
    # SI NO ESTAMOS TRACKING → DETECTAR
    # -------------------------
    if not tracking:

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 800:
                x, y, w, h = cv2.boundingRect(c)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))

                tracking = True

    # -------------------------
    # SI ESTAMOS TRACKING → SEGUIR
    # -------------------------
    else:
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cx = int(x + w/2)
            cy = int(y + h/2)

            cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)

        else:
            # Si pierde el objeto → volver a detectar
            tracking = False
            tracker = None

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask Roja", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()