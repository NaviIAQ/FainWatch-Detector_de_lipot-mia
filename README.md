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

## ¿Qué analiza?

<details>
<summary>Calibración automática</summary>

Al inicio de cada sesión el sistema aprende los valores normales de ese donante concreto: su color de piel en frente y mejillas, su apertura ocular habitual y su postura de reposo. Esto permite que los umbrales se adapten a cada persona, evitando falsos positivos por diferencias físicas entre donantes.

</details>

<details>
<summary>Parámetros monitorizados</summary>

- Cierre de ojos — detecta somnolencia prolongada
- Apertura de boca — distingue malestar de bostezo
- Inclinación de cabeza — roll y pitch respecto a la postura de reposo
- Caída postural — posición de la nariz respecto a los hombros
- Palidez — desviación relativa del color de piel respecto al baseline
- Expresión de malestar — combinación de gestos faciales involuntarios
- Mirada fija — ojos muy abiertos con iris inmóvil sostenido
- Bostezos — registrados en el historial pero sin efecto en el nivel de riesgo

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
cd carpeta-donde-este-el-archivo
python3 fainwatch_local.py
```

Pulsa **Q** en la ventana de la cámara para parar y generar el informe.

### Versión Google Colab

Para usarlo sin instalar nada, abre el archivo en Google Colab. Requiere cuenta de Google y acceso a cámara web desde el navegador.

### Configuración

Todos los parámetros están al final del archivo bajo `Config(...)` con comentarios explicando qué hace cada uno y cómo ajustarlo.

<details>
<summary>Limitaciones</summary>

- Requiere buenas condiciones de iluminación para el análisis de color de piel
- La versión Colab tiene mayor latencia que la versión local
- No detecta lipotimia en donantes fuera del encuadre de la cámara
- No ha sido validado clínicamente

</details>

## Desarrollo

Este proyecto ha sido desarrollado con el apoyo de inteligencia artificial (Claude, de Anthropic) como asistente en la escritura, estructuración y depuración del código.
