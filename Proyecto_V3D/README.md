# Proyecto V3D - Fruit Ninja con Tracking de Objeto

## Descripción
Este proyecto implementa un juego interactivo estilo "Fruit Ninja" controlado mediante la detección de movimiento de un objeto físico (por defecto de color azul/cyan) a través de una webcam. 

La arquitectura divide el sistema en dos procesos independientes que se comunican mediante **UDP**:
1.  **Visión Artificial**: Procesa la imagen de la cámara, corrige la distorsión de la lente y rastrea el objeto.
2.  **Juego (PyGame)**: Renderiza los gráficos y la lógica del juego, recibiendo las coordenadas del objeto como "cursor".

Esta separación permite ejecutar el juego en un proyector mientras se monitoriza la cámara en otra pantalla.

---

## Requisitos
Asegúrate de tener instalado Python 3.x y las siguientes librerías:

```bash
pip install opencv-python numpy pygame
```

---

## Estructura de Archivos

###  Archivos Principales (Ejecutables)

*   **`Tracker_Palo.py`**  
    *   **Qué hace:** Es el "driver" del sistema. Captura la imagen de la webcam, busca si existe un archivo de calibración (`camera_calibration.npz`) para corregir el efecto ojo de pez, detecta el color configurado (HSV) y envía la posición (X, Y) al juego por el puerto 5005.
    *   **Uso:** Debe ejecutarse siempre en paralelo al juego.

*   **`game.py`**  
    *   **Qué hace:** El juego principal. Lanza la ventana gráfica, genera frutas y bombas, y escucha por UDP las coordenadas para simular el corte.
    *   **Argumentos:**
        *   `--projection`: Cambia el fondo a negro absoluto (ideal para proyectores).
        *   `--fullscreen`: Inicia en pantalla completa.

*   **`calibracion.py`**  
    *   **Qué hace:** Herramienta interactiva para calibrar tu cámara. Te pide que muestres un tablero de ajedrez en diferentes ángulos para calcular la matriz de corrección y eliminar la distorsión de la lente.

###  Archivos de Soporte (Módulos)

*   **`input_handler.py`**:  
    Clase que gestiona la recepción de paquetes UDP en un hilo separado para no bloquear el juego.
*   **`entities.py`**:  
    Define las clases de los objetos: `FlyingShape` (frutas/bombas), `SplitHalf` (mitades cortadas) y `SlashParticle` (efectos visuales).
*   **`utils.py`**:  
    Funciones matemáticas utilitarias para detectar colisiones entre el segmento de corte y los círculos (frutas).
*   **`entorno.py`**:  
    Script alternativo/legacy para lanzar el juego con argumentos básicos.
*   **`Tracker_Objeto.py`**:  
    Versión antigua del tracker. Sirve como referencia simple de detección de colores (rojo/amarillo) sin conexión UDP ni calibración avanzada.

---

## Guía de Ejecución

Para jugar, necesitas abrir **dos terminales** diferentes.

### Paso 1: Calibración (Opcional pero recomendado)
Si tu cámara tiene efecto "ojo de pez" o deforma mucho la imagen:
1.  Imprime un tablero de ajedrez (patrón 9x6 esquinas internas).
2.  Ejecuta:
    ```bash
    python Proyecto_V3D/calibracion.py
    ```
3.  Sigue las instrucciones en pantalla (`s` para capturar, `c` para calibrar). Esto generará el archivo `camera_calibration.npz`.

### Paso 2: Ejecutar el Juego
En la **Terminal 1**, inicia el juego a la espera de datos:

**Modo Desarrollo (Ventana normal):**
```bash
python Proyecto_V3D/game.py
```

**Modo Proyección (Pantalla completa y fondo negro):**
```bash
python Proyecto_V3D/game.py --projection --fullscreen
```

### Paso 3: Ejecutar el Tracker
En la **Terminal 2**, inicia la visión artificial:
```bash
python Proyecto_V3D/Tracker_Palo.py
```
*Se abrirá una ventana mostrando lo que ve la cámara. Asegúrate de que tu objeto (palo azul) sea detectado y tenga un círculo verde encima.*

---

## Configuración de Color

El tracker está configurado por defecto para detectar tonos **azules/cyan**. Si usas otro objeto, edita `Tracker_Palo.py` y modifica los rangos HSV:

```python
# Ejemplo para color AZUL/CYAN
lower_color = np.array([90, 100, 100])
upper_color = np.array([130, 255, 255])
```
