# Proyecto V3D - Fruit Ninja AR en 3D con Homografía y Tracking

## Descripción
Este proyecto implementa un sistema de Realidad Aumentada (AR) y un motor 3D interactivo estilo Fruit Ninja.
El jugador controla el sable cortante mediante un objeto físico (palo/marcador de color) en el mundo real que es rastreado por una webcam.

El sistema usa **Homografía** para mapear una superficie de juego física (como una mesa) a un entorno 3D (*Ursina Engine*). Todo esto interactúa bidireccionalmente, de tal forma que los objetos tridimensionales cortados, las bombas y todos los sistemas de partículas se proyectan isométrica y volumétricamente en la imagen en vivo de tu cámara como auténtica AR.

## Arquitectura (Bidireccional por UDP)
El sistema ejecuta dos procesos sincrónicos de baja latencia:

1. **Tracker AR** (`Tracker_Palo.py`) - Envía entradas al puerto `5005` y escucha el estado en `5006`
- Procesa el feed de vídeo y rastrea tu espada/marcador por color (HSV).
- Aplica corrección y **Homografía** (calibración extrínseca) para mapear tu mesa -> plano virtual.
- Renderiza isométrica y volumétricamente los polígonos del entorno 3D sobre la cámara, adaptando el campo de visión (FOV) real de la cámara.
- Dibuja la UI completa en Realidad Aumentada (Vidas, Score, Combos, Estela del arma).

2. **Juego 3D** (`game_3d.py`) - Envía red a `5006` y escucha inputs en `5005`
- Ejecuta toda la lógica del juego impulsada por *Ursina Engine*.
- Formas geométricas volumétricas y cálculos físicos (Prismas, Cilindros, Octaedros y Bombas Dodecaedro).
- **Mecánicas avanzadas:** Frutas especiales "Duras" que requieren múltiples cortes, partículas tridimensionales que rebotan, sistema de combos dinámico, ondas expansivas al cortar bombas, cortes que respetan el arco de partición.
- Renderiza sombras, iluminación direccional y proyecta las frutas cortadas con motor de físicas exacto. *(Mantiene `game.py` como versión ligera 2D original)*.

---

## Mecánicas del Juego
- **Frutas Regulares:** Simulan formas geométricas, se parten en dos siguiendo la dirección y ángulo exacto del corte.
- **Frutas Especiales/Duras:** Un 12% de probabilidad que requieran de 5 a 9 cortes. Al golpearlas entran en un estado alterado de tiempo casi congelado (*Matrix-style*) y emiten destellos.
- **Bombas:** Renderizadas como dodecaedros negros con bordes rojos. Cortarlas provoca vibración de cámara, pérdida del combo, una vida y expulsa partículas rojas.
- **Combos:** Realizar cortes seguidos otorga puntos extra y aparece un feedback de "+{X} COMBO!".

---

## Calibración de cámara: 2 scripts, 2 archivos .npz
Para calibración intrínseca se usan dos scripts distintos:

1. `calibracion.py` -> Clásico con tablero de ajedrez (genera `camera_calibration.npz`).
2. `calibracion_charuco.py` -> Tablero ChArUco (genera `camera_charuco_calibration.npz`).

Puedes configurar cuál usar editando la variable `calibration_file` dentro de `Tracker_Palo.py`.

---

## Calibración de superficie (Homografía en Vivo)
Para que el cursor físico llegue correctamente al espacio 3D real:
1. Haz click izquierdo 4 veces en tu ventana de tracker en la zona donde desees jugar de la vida real.
2. Sigue este orden de dirección horaria: **Superior-Izquierda -> Superior-Derecha -> Inferior-Derecha -> Inferior-Izquierda**.
3. El motor asimilará esto como la ventana por donde viajarán los cuerpos voladores en Realidad Aumentada.

**Controles del Tracker:**
- `R`: Reiniciar la cámara y solicitar los 4 puntos de homografía de nuevo.
- `Q`: Salir del servicio.

---

## Guía rápida de uso (Arranque Dual)

Es necesario ejecutar **ambos scripts al mismo tiempo** para sincronizar entorno virtual y entorno real.

1. **Ejecutar Tracker de Realidad Aumentada:**
```bash
python Proyecto_V3D/Tracker_Palo.py
```
*(Haz la calibración de tu espacio en la mesa u hoja en ese momento con los 4 clicks).*

2. **Ejecutar el Entorno Físico 3D:**
```bash
python Proyecto_V3D/game_3d.py
```
*(Puedes usar `python Proyecto_V3D/game_3d.py --no-udp` si quieres probar el comportamiento del juego usando sólo el ratón local del sistema, útil en desarrollo).*

---

## Características Clave (Logros)
- Extracción de formas 3D nativas calculadas a través del modelo de cámara (FOV/Z/World-Size).
- AR avanzada que dibuja los polígonos correspondientes exactamente con la geometría rotacional XYZ enviada desde Python hacia el display con OpenCV en vivo.
- Interpolación entre el trazado y dibujado continuo al rellenar espacios muertos durante movimientos bruscos y rápidos del palo real.

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

