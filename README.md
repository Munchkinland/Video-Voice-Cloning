Video Voice Cloning

Esta aplicación permite la transcripción, traducción y clonación de voz en videos mediante el uso de Whisper de OpenAI para la transcripción, noisereduce para la reducción de ruido, y tecnologías de Text-to-Speech (TTS) para mantener la voz original del hablante en las traducciones.
Funcionalidades

    Transcripción automática de videos en varios idiomas utilizando Whisper.
    Traducción de transcripciones a múltiples idiomas.
    Clonación de voz: Genera audios traducidos con la voz original del hablante.
    Reducción de ruido en el audio mediante noisereduce.
    Procesamiento paralelo con multihilos para optimizar el rendimiento.

Requisitos

    Python 3.8+
    ffmpeg para la manipulación de archivos de audio y video.

Instalación de ffmpeg

Puedes instalar ffmpeg de la siguiente manera:

    En Windows: Descarga desde este enlace e incluye la ruta del ejecutable en tu variable PATH.

Uso

    Transcribir y traducir un video:

    Para transcribir y traducir el video a un idioma específico:

    bash

python transcribe_translate.py --input ./ruta_al_video.mp4 --language es --output ./ruta_de_salida

Donde:

    --input: Ruta al archivo de video.
    --language: Idioma de salida (es para español, en para inglés, etc.).
    --output: Carpeta de salida para la transcripción y el audio.

Aplicar reducción de ruido (opcional):

bash

    python transcribe_translate.py --input ./ruta_al_video.mp4 --language es --output ./ruta_de_salida --reduce-noise

    Parámetros adicionales:
        --model: Especifica el modelo Whisper a utilizar (small, base, large).
        --threads: Número de hilos para el procesamiento en paralelo (por defecto 4).
        --segment-length: Duración de los segmentos de audio (por defecto 30 segundos).

Estructura del Proyecto

markdown

.
├── README.md

├── requirements.txt

├── transcribe_translate.py

├── utils

│   ├── audio_processing.py

│   ├── voice_cloning.py

│   ├── whisper_integration.py

└── outputs

    ├── transcriptions
    
    └── audios

Contribución

Si quieres contribuir al proyecto, puedes abrir un issue o enviar un pull request.
Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
