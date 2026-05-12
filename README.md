# FainWatch-Detector_de_lipotimia
Sistema de monitorización facial para donantes de sangre. Analiza en tiempo real la postura, el color de piel, la apertura ocular y la expresión facial mediante una cámara estándar y sin contacto con el donante. Podría permitir al personal detectar de forma precoz los signos previos a una lipotimia y actuar antes de que ocurra el episodio.

## Información de uso

> [!IMPORTANT]
> ### Requisitos
> - Cuenta de Google
> - Google Colab (el código está diseñado específicamente para esta plataforma)
> - Navegador con acceso a cámara web

> [!TIP]
> ### Cómo usarlo
> 1. Abre el archivo en Google Colab
> 2. Ejecuta la celda (Ctrl + F9)
> 3. Acepta el permiso de cámara en el navegador
> 4. Pulsa INICIAR

> [!NOTE]
> ### Dependencias que se instalan automáticamente
> - mediapipe
> - opencv-python-headless
> - Pillow
> - reportlab

> [!WARNING]
> ### Adaptación a otras plataformas
> El código usa google.colab para la captura de cámara y los botones.
> Si quieres ejecutarlo en local o como aplicación web, necesitaría
> adaptarse para usar OpenCV nativo (local) o Streamlit (web).

## ¿Qué hace FainWatch?

FainWatch es una herramienta de monitorización facial desarrollada en Python para Google Colab. Utiliza la cámara del ordenador para analizar en tiempo real el rostro y la postura del donante durante la extracción de sangre, con el objetivo de detectar de forma precoz los signos previos a una lipotimia vasovagal.

<details>
<summary>Calibración automática</summary>

Al inicio de cada sesión, durante los primeros 50 frames (~4 segundos), el sistema aprende los valores normales de ese donante concreto: su color de piel en frente y mejillas, su apertura ocular habitual y su postura de reposo. Esto permite que los umbrales de detección se adapten a cada persona, evitando falsos positivos por diferencias físicas entre donantes.

</details>

<details>
<summary>Qué analiza</summary>

A partir de la calibración, monitoriza de forma continua ocho parámetros:
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
- historico_sesion.csv — todos los datos frame a frame
- informe_sesion.pdf — resumen con línea de tiempo, alertas y métricas medias

</details>

<details>
<summary>Limitaciones</summary>

FainWatch es un prototipo funcional desarrollado como proyecto piloto. No ha sido validado clínicamente. Requiere Google Colab para funcionar y una cámara web con buenas condiciones de iluminación para obtener resultados fiables.

</details>

<details>
<summary>Desarrollo</summary>

Esta herramienta ha sido desarrollada con el apoyo de inteligencia artificial como asistente en la escritura y estructuración del código.

</details>
