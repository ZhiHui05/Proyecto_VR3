# Proyecto V3D - Fruit Ninja AR con Homografia y Tracking

## Descripcion
Este proyecto implementa un sistema de Realidad Aumentada (AR) interactivo estilo Fruit Ninja.
El jugador controla el sable mediante un objeto fisico de color azul/cyan rastreado por webcam.

El sistema usa Homografia para mapear una superficie real (mesa) a la pantalla del juego,
permitiendo una interaccion precisa en toda el area y proyectando los objetos virtuales
sobre la imagen de la camara.

## Arquitectura (Bidireccional por UDP)
El sistema tiene dos procesos en tiempo real:

1. Tracker (Tracker_Palo.py) - envia al puerto 5005
- Procesa la imagen de camara.
- Aplica calibracion intrinseca (correccion de lente) si hay archivo .npz cargado.
- Aplica homografia (calibracion extrinseca) para transformar coordenadas camara -> juego.
- Envia la posicion del cursor (X, Y) al juego.
- Recibe estado del juego y dibuja frutas/bombas sobre el video (AR).

2. Juego (game.py) - envia al puerto 5006
- Ejecuta logica de juego (fisica, puntuacion, vidas).
- Envia estado de entidades al tracker para renderizado AR.
- Muestra cursor visual sincronizado con el tracking.

---

## Calibracion de camara: 2 scripts, 2 archivos .npz
Para calibracion intrinseca se han usado dos scripts distintos:

1. calibracion.py
- Metodo clasico con patron de tablero.
- Genera: camera_calibration.npz

2. calibracion_charuco.py
- Metodo con tablero ChArUco (mas robusto en algunos escenarios).
- Genera: camera_charuco_calibration.npz

Importante:
- No se usan los dos .npz a la vez.
- En Tracker_Palo.py se carga solo uno, segun la opcion que quieras utilizar.

### Como elegir la calibracion en Tracker_Palo.py
En run_tracker(), cambia el valor de calibration_file:

```python
# Opcion 1 (calibracion clasica)
calibration_file = "camera_calibration.npz"

# Opcion 2 (calibracion Charuco)
# calibration_file = "camera_charuco_calibration.npz"
```

Puedes dejar activa solo una de las dos opciones.

---

## Calibracion de superficie (Homografia)
Para que el cursor llegue correctamente a toda la pantalla:
- Define 4 puntos manualmente con click izquierdo en la ventana del tracker.
- Orden: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.
- Con esos puntos se calcula la matriz de homografia.

Controles:
- R: reinicia la homografia (vuelve a pedir 4 puntos).
- Q: salir.

---

## Guia rapida de uso
1. Ejecutar juego:

```bash
python Proyecto_V3D/game.py
```

Opcional pantalla completa:

```bash
python Proyecto_V3D/game.py --fullscreen
```

2. Ejecutar tracker:

```bash
python Proyecto_V3D/Tracker_Palo.py
```

3. En la ventana del tracker:
- Seleccionar 4 esquinas de la zona real de juego.
- Verificar que el cursor responde en toda el area.

---

## Caracteristicas clave
- Tracking por color HSV (azul/cyan).
- Comunicacion UDP de baja latencia.
- Homografia para corregir perspectiva en mesa inclinada.
- Renderizado AR de frutas y bombas sobre imagen real.
- Sincronizacion de mitades de fruta tras cortes.

---

## Solucion de problemas
- Si no encuentra calibracion:
    revisa que el .npz elegido exista en la ruta esperada.

- Si hay lag:
    asegurate de ejecutar tracker y juego en la misma maquina (localhost).

- Si el color no se detecta bien:
    mejora iluminacion o ajusta el rango HSV en Tracker_Palo.py.

---

## Estructura principal
- Tracker_Palo.py: tracking, homografia y render AR.
- game.py: motor de juego.
- calibracion.py: calibracion intrinseca clasica.
- calibracion_charuco.py: calibracion intrinseca con ChArUco.
- camera_calibration.npz: salida de calibracion.py.
- camera_charuco_calibration.npz: salida de calibracion_charuco.py.
- entities.py y utils.py: entidades y utilidades.

