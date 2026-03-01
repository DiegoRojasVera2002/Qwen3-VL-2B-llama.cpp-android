#!/usr/bin/env python3
"""
PathsAI Termux — Asistente visual para personas invidentes en Android.

Usa herramientas nativas de Termux:API (sin OpenCV, sin pyttsx3):
  - termux-camera-photo  → captura frames
  - termux-tts-speak     → TTS nativo Android en español
  - termux-wake-lock     → evita que el celular duerma

Requisitos en Termux:
  pkg install python termux-api
  pip install requests

Uso:
  python paths_ai_termux.py [--interval 3] [--url http://localhost:8080] [--camera 0]
"""

import argparse
import base64
import signal
import subprocess
import sys
import tempfile
import threading
import time
import os

import requests

PROMPT = (
    "Eres PathsAI, los ojos de una persona ciega que está caminando. "
    "Analiza la imagen y responde SIEMPRE en español con máximo 3 frases cortas y directas. "
    "Tu respuesta debe tener SIEMPRE esta estructura: "
    "PRIMERO describe brevemente qué hay adelante (entorno general). "
    "SEGUNDO si hay obstáculos, personas o peligros indica qué es, dónde está "
    "(izquierda, derecha, adelante) y qué tan cerca (muy cerca <1m, cerca 1-2m, a distancia >3m). "
    "TERCERO da una instrucción de movimiento directa: Gira a la izquierda, Desvíate a la derecha, "
    "Detente, Continúa adelante, Cambia de ruta. "
    "Si no hay obstáculos ni peligros: Camino despejado, continúa adelante."
)

running = True
is_speaking = threading.Event()


def signal_handler(_sig, _frame):
    global running
    print("\n[info] Deteniendo PathsAI...")
    running = False


def capture_frame(camera_id: int) -> bytes | None:
    """Captura un frame usando termux-camera-photo."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["termux-camera-photo", "-c", str(camera_id), tmp_path],
            timeout=10,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"[warn] Camera error: {result.stderr.decode()}", file=sys.stderr)
            return None

        with open(tmp_path, "rb") as f:
            return f.read()
    except subprocess.TimeoutExpired:
        print("[warn] Timeout capturando frame.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[error] capture_frame: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def analyze_frame(jpeg_bytes: bytes, server_url: str) -> str | None:
    """Envía frame al llama-server y retorna la respuesta."""
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "max_tokens": 80,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.ConnectionError:
        return None
    except Exception as e:
        print(f"[error] analyze_frame: {e}", file=sys.stderr)
        return None


def speak(text: str) -> None:
    """Habla el texto usando termux-tts-speak (TTS nativo Android)."""
    is_speaking.set()
    try:
        subprocess.run(
            ["termux-tts-speak", "-l", "es", "-r", "1.0", text],
            timeout=30,
        )
    except Exception as e:
        print(f"[error] TTS: {e}", file=sys.stderr)
    finally:
        is_speaking.clear()


def speak_async(text: str) -> None:
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()


def main_loop(server_url: str, interval: float, camera_id: int) -> None:
    global running

    print("[info] Iniciando PathsAI en Termux...")
    print(f"[info] Servidor: {server_url} | Intervalo: {interval}s | Cámara: {camera_id}")
    print("[info] Presiona Ctrl+C para detener.\n")

    speak_async("PathsAI iniciado. Analizando entorno.")

    last_response = ""

    while running:
        if is_speaking.is_set():
            time.sleep(0.2)
            continue

        jpeg = capture_frame(camera_id)
        if jpeg is None:
            time.sleep(2)
            continue

        text = analyze_frame(jpeg, server_url)

        if text is None:
            if last_response != "__server_down__":
                speak_async("Servidor no disponible.")
                last_response = "__server_down__"
            time.sleep(interval)
            continue

        if text and text != last_response:
            print(f"[ai] {text}")
            speak_async(text)
            last_response = text

        time.sleep(interval)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="PathsAI Termux - Asistente visual para personas invidentes")
    parser.add_argument("--interval", type=float, default=3.0, help="Segundos entre análisis (default: 3)")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="URL del servidor llama.cpp")
    parser.add_argument("--camera", type=int, default=0, help="ID de cámara: 0=trasera, 1=frontal (default: 0)")
    args = parser.parse_args()

    main_loop(args.url, args.interval, args.camera)
    print("[info] PathsAI detenido.")


if __name__ == "__main__":
    main()
