import cv2
import numpy as np
import argparse
import os

# -----------------------------
# Configuración del Tablero ChArUco
# -----------------------------
# Valores ajustados a tu tablero físico
SQUARES_X = 11        # Número de cuadros en X (Columnas)
SQUARES_Y = 8         # Número de cuadros en Y (Filas)
SQUARE_LENGTH = 0.015 # Longitud del cuadrado en metros
MARKER_LENGTH = 0.011 # Longitud del marcador en metros
DICTIONARY_ID = cv2.aruco.DICT_4X4_250

def get_charuco_setup():
    """Configura y devuelve el tablero y el detector."""
    # OpenCV 4.7+
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    try:
        detector = cv2.aruco.CharucoDetector(board)
    except AttributeError:
        # Fallback para versiones más antiguas si fuera necesario, aunque el usuario tiene la nueva
        print("Advertencia: cv2.aruco.CharucoDetector no encontrado. Asegúrate de tener opencv-contrib-python >= 4.7.0")
        detector = None 
    return board, detector

# -----------------------------
# Captura y Calibración
# -----------------------------
def calibrate_camera_charuco(num_images=15, camera_id=1):
    
    board, detector = get_charuco_setup()
    if detector is None:
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con ID {camera_id}")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    
    captured = 0
    img_size = None

    print("\n--------------------------------------")
    print(f"Objetivo: Capturar {num_images} imágenes")
    print("CONTROLES:")
    print("  's' -> Guardar frame actual (si es válido)")
    print("  'q' -> Salir / Terminar captura")
    print("--------------------------------------\n")

    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Detección
        # Nota: detectBoard devuelve (charucoCorners, charucoIds, markerCorners, markerIds)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        display = frame.copy()
        valid_frame = False

        # Dibujar resultados si se encuentran
        if charuco_corners is not None and len(charuco_corners) > 4:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            valid_frame = True
        
        # Información en pantalla
        color = (0, 255, 0) if valid_frame else (0, 0, 255)
        cv2.putText(display, f"Capturadas: {captured}/{num_images}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if valid_frame:
            cv2.putText(display, "Detectado! Presiona 's'", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibracion Charuco", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if valid_frame:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                captured += 1
                print(f"Imagen {captured} guardada.")
            else:
                print("No se detectaron suficientes esquinas para guardar este frame.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Proceder a calibrar si hay suficientes datos
    if len(all_charuco_corners) < 3:
        print("\nNo se capturaron suficientes frames para una calibración confiable.")
        return

    print("\nCalculando calibración...")
    
    try:
        # Intentar usar calibrateCameraCharuco si está disponible (OpenCV Contrib)
        if hasattr(cv2.aruco, 'calibrateCameraCharuco'):
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_charuco_corners,
                charucoIds=all_charuco_ids,
                board=board,
                imageSize=img_size,
                cameraMatrix=None,
                distCoeffs=None
            )
        else:
            # Fallback: Usar calibración estándar si la función específica no está disponible
            print("Función cv2.aruco.calibrateCameraCharuco no encontrada. Usando calibración estándar...")
            
            # Preparar puntos 3D y 2D
            objpoints = [] 
            imgpoints = [] 
            
            # Obtener todas las esquinas del tablero (puntos 3D)
            # Nota: En versiones recientes, board.getChessboardCorners() devuelve los puntos.
            all_board_corners = board.getChessboardCorners()

            for i in range(len(all_charuco_corners)):
                current_corners = all_charuco_corners[i]
                current_ids = all_charuco_ids[i]
                
                if current_corners is None or len(current_corners) == 0:
                    continue
                
                # Filtrar puntos 3D correspondientes a los IDs detectados
                current_obj_points = []
                for id_val in current_ids.flatten():
                    current_obj_points.append(all_board_corners[id_val])
                
                objpoints.append(np.array(current_obj_points, dtype=np.float32))
                imgpoints.append(current_corners)

            retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, 
                imgpoints, 
                img_size, 
                None, 
                None
            )

        print("\n--------------------------------------")
        print("RESULTADOS DE CALIBRACION")
        print("--------------------------------------")
        print(f"Error de Reproyección: {retval}")
        print("\nCamera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs)

        # Guardar en el mismo formato que calibracion.py para compatibilidad con otros scripts
        output_file = "camera_charuco_calibration.npz"
        np.savez(output_file, 
                 mtx=camera_matrix, 
                 dist=dist_coeffs, 
                 rvecs=rvecs, 
                 tvecs=tvecs)
        
        print(f"\nCalibración guardada exitosamente en '{output_file}'")

    except Exception as e:
        print(f"\nError durante el cálculo de calibración: {e}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Script de calibración usando tablero ChArUco")
    
    parser.add_argument(
        "--images",
        type=int,
        default=15,
        help="Número de imágenes a capturar para la calibración"
    )

    parser.add_argument(
        "--cam",
        type=int,
        default=1,
        help="ID de la cámara a utilizar (p.ej. 0 para interna, 1 para externa)"
    )

    args = parser.parse_args()

    calibrate_camera_charuco(num_images=args.images, camera_id=args.cam)

if __name__ == "__main__":
    main()
