# FainWatch — Detector de Lipotimia

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Estado-Prototipo-orange)
![Platform](https://img.shields.io/badge/Platform-macOS%20|%20Windows%20|%20Linux-lightgrey)

Sistema de monitorización facial para donantes de sangre que analiza en tiempo real la postura, el color de piel, la apertura ocular y la expresión facial mediante una cámara estándar, sin contacto con el donante. Podría permitir detectar de forma precoz los signos previos a una lipotimia y actuar antes de que ocurra el episodio.

## Objetivo

La lipotimia vasovagal es uno de los eventos adversos más frecuentes en los procesos de donación de sangre. Su aparición brusca, combinada con la imposibilidad de monitorizar a todos los donantes simultáneamente, hace que la detección temprana dependa de la observación visual directa o de que el propio donante comunique su malestar, algo que no siempre ocurre a tiempo.

FainWatch nace como una idea personal para explorar si es posible detectar estos episodios de forma automática y no invasiva.

## Fase actual

> [!WARNING]
> FainWatch es un prototipo funcional en fase piloto. No ha sido validado clínicamente. Los resultados no deben usarse como base para decisiones médicas sin supervisión profesional.

El sistema detecta y analiza señales en tiempo real pero sus umbrales no han sido contrastados con episodios reales de lipotimia. El siguiente paso es recoger datos etiquetados de sesiones reales para validar y mejorar la detección.

## Requisitos
- Python 3.8 o superior
- Webcam integrada o externa
- Buena iluminación ambiental
- macOS, Windows o Linux
- Los modelos de MediaPipe (descargar y colocar en la misma carpeta que el código):
  - [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
  - [pose_landmarker.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)

## ¿Qué analiza?

<details>
<summary>Calibración automática</summary>

Al inicio de cada sesión el sistema aprende los valores normales de ese donante concreto: su color de piel en frente y mejillas, su apertura ocular habitual y su postura de reposo. Esto permite que los umbrales se adapten a cada persona, evitando falsos positivos por diferencias físicas entre donantes.

</details>

<details>
<summary>Parámetros monitorizados</summary>

| Parámetro | Qué detecta | Puntos de riesgo |
|-----------|-------------|-----------------|
| Cierre de ojos | Somnolencia prolongada (blendshapes) | +3 |
| Caída postural | Nariz cae respecto a los hombros | +3 |
| Mirada fija | EAR elevado + iris inmóvil ≥22 frames | +3 |
| Palidez cutánea | Desviación de color respecto al baseline personal | +2 |
| Cabeza lateral | Roll fuera del rango calibrado | +2 |
| Boca abierta | MAR elevado (distingue malestar de bostezo) | +1 |
| Malestar | 11 blendshapes faciales involuntarios combinados | +1 |
| Bostezos | MAR alto + ojos entrecerrados simultáneos | solo historial |

> - El nivel de riesgo se activa cuando hay ≥2 señales simultáneas activas.
> - La puntuación total determina el nivel: 1-2 pts = Precaución, ≥3 pts = Alerta.
> - El nivel final es la media de los últimos 15 frames para evitar falsos positivos.

</details>

<details>
<summary>Filtros de movimiento voluntario</summary>

El sistema distingue entre movimiento voluntario y síntoma clínico. Si el donante gira la cabeza para hablar con alguien o mira hacia un lado, el sistema lo ignora analizando la velocidad del movimiento, la coherencia entre parámetros y la duración sostenida de cada señal.

</details>

<details>
<summary>Niveles de riesgo</summary>

Combina los parámetros en un sistema de puntuación y muestra en pantalla tres estados:
- Normal — donante estable
- Precaución — señales leves presentes
- Alerta — intervención recomendada

</details>

<details>
<summary>Historial de sesión</summary>

Al finalizar cada sesión genera automáticamente:
- `historico_sesion.csv` — todos los datos frame a frame
- `informe_sesion.pdf` — resumen con línea de tiempo, alertas y métricas medias

</details>

## Uso

### Versión local (recomendada)

Requiere Python 3.8 o superior y una webcam. Las dependencias se instalan automáticamente la primera vez.

```bash
cd ~/Desktop/FainWatch
python3 fainwatch_local.py
```

Pulsa **Q** en la ventana de la cámara para parar y generar el informe.

### Configuración

Todos los parámetros están al final del archivo bajo `Config(...)` con comentarios explicando qué hace cada uno y cómo ajustarlo.

<details>
<summary>Limitaciones</summary>

- Requiere buenas condiciones de iluminación para el análisis de color de piel
- No detecta lipotimia en donantes fuera del encuadre de la cámara
- No ha sido validado clínicamente

</details>

## Desarrollo

Este proyecto ha sido desarrollado con el apoyo de inteligencia artificial (Claude, de Anthropic) como asistente en la escritura, estructuración y depuración del código.

<br> <br> 

<div align="center">
  <h3>Demostracion de Analisis de Secuencia FainWatch</h3>
  <video src="https://github.com/user-attachments/assets/735cbbb5-f0e0-4bb9-9eac-e4bccf74b7c4" width="600" autoplay loop muted playsinline>
  </video>
</div>

El vídeo muestra el análisis en tiempo real sobre una secuencia de 76 segundos generada con imágenes sintéticas de IA, que simulan la progresión de un episodio vasovagal desde el estado normal hasta la lipotimia. Las anotaciones en pantalla muestran el nivel de riesgo actual (Normal / Precaución / Alerta), las alertas activas en ese momento y los valores de cada métrica analizada frame a frame.

## Validación visual

La siguiente gráfica muestra la comparativa entre la valoración clínica observacional y la detección automática del sistema sobre un episodio simulado de 76 segundos (video anterior). 
> La validación se ha realizado sobre imágenes sintéticas generadas con IA. No constituye evidencia clínica.

<img width="1710" height="1037" alt="FainWatch_Comparativa_valoración_clínica_vs_valoración_automática" src="https://github.com/user-attachments/assets/70a55cbf-691c-43f6-9985-1458e62b2c93" />

