#!/usr/bin/env python3
"""
Paths AI MVP - Asistente visual para personas invidentes.

Captura webcam → SmolVLM-500M (llama.cpp) → pyttsx3 (voz).

Uso:
    python paths_ai_mvp.py [--interval 3] [--url http://localhost:8080] [--lang es]
"""

import argparse
import base64
import json
import signal
import sys
import threading
import time

import cv2
import pyttsx3
import requests

PROMPT = "¿Qué obstáculos hay? Responde en español, 2 frases máximo."

# Globals for clean shutdown
running = True
tts_engine = None
tts_lock = threading.Lock()
is_speaking = threading.Event()


def init_tts(lang: str) -> pyttsx3.Engine:
    """Initialize pyttsx3 engine with the given language."""
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    # Find a matching voice for the requested language
    for voice in voices:
        if lang in voice.id:
            engine.setProperty("voice", voice.id)
            break
    engine.setProperty("rate", 145)
    engine.setProperty("volume", 1.0)
    return engine


def capture_frame(cap: cv2.VideoCapture) -> bytes | None:
    """Capture a frame from the webcam, resize, and encode as JPEG bytes."""
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (480, 360))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def analyze_frame(jpeg_bytes: bytes, server_url: str) -> str | None:
    """Send frame to llama.cpp server and return the text response."""
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "max_tokens": 60,
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
        data = resp.json()
        print("\n── respuesta del servidor ──────────────────────────")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("────────────────────────────────────────────────────\n")
        return data["choices"][0]["message"]["content"]
    except requests.ConnectionError:
        return None
    except Exception as e:
        print(f"[error] Análisis falló: {e}", file=sys.stderr)
        return None


def speak(text: str) -> None:
    """Speak text using pyttsx3 in the current thread (called from TTS thread)."""
    global tts_engine
    is_speaking.set()
    try:
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    finally:
        is_speaking.clear()


def speak_async(text: str) -> None:
    """Launch TTS in a separate thread so it doesn't block the main loop."""
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()


def main_loop(server_url: str, interval: float) -> None:
    """Main capture → analyze → speak loop."""
    global running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] No se pudo abrir la webcam.", file=sys.stderr)
        sys.exit(1)

    print("[info] Webcam abierta. Iniciando análisis...")
    speak_async("Paths AI iniciado. Analizando entorno.")

    last_response = ""

    try:
        while running:
            # Don't capture a new frame while still speaking
            if is_speaking.is_set():
                time.sleep(0.1)
                continue

            jpeg = capture_frame(cap)
            if jpeg is None:
                print("[warn] No se pudo capturar frame.", file=sys.stderr)
                time.sleep(1)
                continue

            text = analyze_frame(jpeg, server_url)

            if text is None:
                # Server unreachable — notify once
                if last_response != "__server_down__":
                    speak_async("Servidor no disponible.")
                    last_response = "__server_down__"
                time.sleep(interval)
                continue

            text = text.strip()
            if text and text != last_response:
                print(f"[ai] {text}")
                speak_async(text)
                last_response = text

            time.sleep(interval)
    finally:
        cap.release()
        print("[info] Webcam liberada.")


def main() -> None:
    global running, tts_engine

    parser = argparse.ArgumentParser(description="Paths AI MVP - Asistente visual para personas invidentes")
    parser.add_argument("--interval", type=float, default=3, help="Segundos entre análisis (default: 3)")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="URL del servidor llama.cpp")
    parser.add_argument("--lang", type=str, default="es", help="Idioma del TTS (default: es)")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def signal_handler(_sig, _frame):
        global running
        print("\n[info] Deteniendo Paths AI...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    print("=== Paths AI MVP ===")
    print(f"Servidor: {args.url}")
    print(f"Intervalo: {args.interval}s")
    print("Presiona Ctrl+C para detener.\n")

    tts_engine = init_tts(args.lang)
    main_loop(args.url, args.interval)

    print("[info] Paths AI detenido.")


if __name__ == "__main__":
    main()
