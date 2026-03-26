# Proyecto V3D - Fruit Ninja AR con Homograf赤a y Tracking

## Descripci車n
Este proyecto implementa un sistema de Realidad Aumentada (AR) interactivo estilo "Fruit Ninja". El jugador controla el "sable" mediante un objeto f赤sico (color azul/cyan) rastreado por una webcam. 

El sistema destaca por utilizar **Homograf赤a** para mapear la superficie de una mesa real a la pantalla del juego, permitiendo una interacci車n precisa incluso en las esquinas, y proyectando los elementos del juego virtual de vuelta sobre la imagen de la c芍mara.

## Arquitectura (Bidireccional)
El sistema consta de dos procesos que se comunican por **UDP** en tiempo real:

1.  **Tracker (`Tracker_Palo.py`) - Puerto 5005 -> Juego**:
    *   Procesa la imagen de la c芍mara.
    *   Aplica correcci車n de lente (calibraci車n intr赤nseca).
    *   Calcula la **Homograf赤a** (calibraci車n extr赤nseca) para transformar coordenadas de c芍mara a juego.
    *   Env赤a la posici車n del cursor $(X, Y)$ al juego.
    *   **Renderiza AR**: Recibe el estado del juego y dibuja frutas/bombas sobre la imagen de la video.

2.  **Juego (`game.py`) - Puerto 5006 -> Tracker**:
    *   Renderiza la l車gica del juego (f赤sica, puntuaci車n, vidas).
    *   Env赤a el estado de todas las "Entidades" (tipo, posici車n, radio, color) al tracker para la visualizaci車n AR.
    *   Muestra un cursor visual (anillo verde) que sigue al objeto rastreado.

---

## Novedades y Caracter赤sticas Clave

### 1. Calibraci車n de Superficie (Homograf赤a)
Para solucionar el problema de no alcanzar las esquinas o la distorsi車n de perspectiva al jugar sobre una mesa inclinada respecto a la c芍mara:
*   El tracker permite definir manualmente las **4 esquinas de la zona de juego**.
*   Esto crea una matriz de transformaci車n que "endereza" la imagen y asigna exactamente el 芍rea f赤sica delimitada a la pantalla completa del juego.

### 2. Realidad Aumentada (AR) S車lida
*   El tracker recibe la posici車n de las frutas y bombas del juego.
*   Utiliza la **matriz inversa de la homograf赤a** para proyectar esos objetos virtuales sobre el video de la c芍mara.
*   **Mejora visual**: Los objetos se renderizan con colores s車lidos (sin transparencia) para evitar que la imagen se vea oscura o lavada.

### 3. Sincronizaci車n de Cortes
*   Cuando cortas una fruta en el juego, las dos mitades resultantes tambi谷n se env赤an al tracker y se visualizan cayendo en la pantalla de la c芍mara.

---

## Gu赤a de Uso

### Paso 1: Iniciar el Juego
Abre una terminal y ejecuta el juego. Este se quedar芍 esperando datos del tracker.

**Ventana normal (Recomendado para pruebas):**
```bash
python Proyecto_V3D/game.py
```

**Pantalla completa (Para jugar):**
```bash
python Proyecto_V3D/game.py --fullscreen
```

### Paso 2: Iniciar el Tracker y Calibrar
Abre una **segunda terminal** y ejecuta el tracker:

```bash
python Proyecto_V3D/Tracker_Palo.py
```

**Proceso de Calibraci車n (IMPORTANTE):**
1.  Se abrir芍 la ventana de la c芍mara.
2.  Haz **CLICK IZQUIERDO** en las 4 esquinas de tu 芍rea de juego real (por ejemplo, las esquinas de tu mesa o alfombrilla) en este orden:
    *   Superior Izquierda -> Superior Derecha -> Inferior Derecha -> Inferior Izquierda.
3.  Al completar los 4 puntos, el sistema calcular芍 la homograf赤a.
4.  ?Listo! Ahora el cursor del juego deber赤a llegar perfectamente a todas las esquinas.

*Nota: Presiona `R` para reiniciar los puntos de calibraci車n o `Q` para salir.*

---

## Soluci車n de Problemas Comunes

*   **"El juego va con lag / retraso"**: 
    *   El tracker incluye un sistema de "drenado de buffer" para procesar siempre el 迆ltimo paquete recibido y descartar los viejos. Aseg迆rate de que ambos scripts corren en la misma m芍quina (localhost).
*   **"Error: UnboundLocalError: local_variable referenced before assignment"**:
    *   Este error ocurr赤a en versiones anteriores cuando el juego no enviaba datos al inicio. Ha sido corregido inicializando variables por defecto (`local_game_width`, `local_game_height`) en el tracker.
*   **"La imagen de la c芍mara se ve muy oscura"**:
    *   Se ha eliminado la transparencia (`cv2.addWeighted`) en la capa de AR. Ahora los objetos se dibujan directamente sobre el frame, manteniendo el brillo original de la c芍mara.

---

## Estructura de Archivos Actualizada

*   `Tracker_Palo.py`: L車gica principal de visi車n, homograf赤a y AR.
*   `game.py`: Motor del juego.
*   `camera_calibration.npz`: Archivo con los datos intr赤nsecos de tu c芍mara (generado por `calibracion.py`).
*   `utils.py`: Funciones auxiliares de geometr赤a.
*   `entities.py`: Clases del juego.

