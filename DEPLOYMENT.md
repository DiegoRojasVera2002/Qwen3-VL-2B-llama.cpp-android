# PathsAI — Guía de Despliegue Completa

> Asistente visual en tiempo real para personas con discapacidad visual.
> Huawei ICT Competition 2026 — Innovation Track

---

## Índice

1. [Arquitectura del Sistema](#arquitectura)
2. [Modelos Investigados](#modelos-investigados)
3. [Despliegue en PC con GPU](#pc-con-gpu)
4. [Despliegue en Android / S25 Ultra (Termux)](#android-termux)
5. [Cómo encender otro día](#encender-otro-dia)
6. [Estructura del Repositorio](#estructura)

---

## Arquitectura

```
Cámara (celular o PC)
        │
        ▼ JPEG frame cada 3s
┌──────────────────────────┐
│  llama-server            │
│  Qwen3-VL-2B (GGUF)     │  ← modelo de visión multilingüe
│  puerto :8080            │
└──────────────────────────┘
        │
        ▼ respuesta en español
┌──────────────────────────┐
│  index.html (Chrome)     │  ← interfaz web + TTS nativo
│  ó paths_ai_termux.py    │  ← script Termux + termux-tts-speak
└──────────────────────────┘
        │
        ▼
  Voz en español describiendo el entorno al usuario
```

---

## Modelos Investigados

| Modelo | Params | Español | GPU | Veredicto |
|--------|--------|---------|-----|-----------|
| SmolVLM-500M | 500M | ❌ Solo inglés | ✅ | Descartado |
| Qwen2-VL-2B | 2B | ✅ | ✅ | Bueno, necesita mmproj |
| **Qwen3-VL-2B** | 2B | ✅ | ✅ | **Usado — mejor calidad** |
| Qwen2.5-VL-3B | 3B | ✅ | ✅ | Alternativa superior |

### Por qué Qwen3-VL-2B
- Multilingüe nativo (español)
- GGUF disponible para llama.cpp
- 2B params — corre en móvil y PC
- Compatible con API OpenAI `/v1/chat/completions`

### Por qué no SmolVLM-500M
- Solo inglés (model card: `"Language(s): English"`)
- Aunque el prompt esté en español, responde en inglés

---

## PC con GPU

### Requisitos
- NVIDIA GPU (probado con RTX 4070 Laptop)
- CUDA 13.0+
- Fedora / Ubuntu / Linux
- llama.cpp compilado con CUDA

### Instalación única (ya hecha)

```bash
# 1. Agregar repo CUDA (Fedora 42)
sudo tee /etc/yum.repos.d/cuda-fedora42.repo << 'EOF'
[cuda-fedora42-x86_64]
name=cuda-fedora42-x86_64
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64
enabled=1
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/D42D0685.pub
EOF

# 2. Instalar CUDA toolkit
sudo dnf install cuda-toolkit-13-0 -y

# 3. Agregar nvcc al PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.zshrc
source ~/.zshrc

# 4. Compilar llama.cpp con CUDA
git clone --depth=1 https://github.com/ggml-org/llama.cpp /tmp/llama-src
PATH=/usr/local/cuda-13.0/bin:$PATH cmake -S /tmp/llama-src -B /tmp/llama-src/build -DGGML_CUDA=ON
cmake --build /tmp/llama-src/build --config Release -j$(nproc) --target llama-server

# 5. Reemplazar binario de Homebrew
sudo cp /tmp/llama-src/build/bin/llama-server /home/linuxbrew/.linuxbrew/bin/llama-server
```

### Correr el servidor (cada vez)

```bash
# Solo para conexiones locales (PC)
llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf --port 8080 -ngl 99 --ctx-size 4096 --parallel 1

# Con acceso desde celular en la misma WiFi
llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf --port 8080 --host 0.0.0.0 -ngl 99 --ctx-size 4096 --parallel 1
```

### Servir el frontend web

```bash
# Terminal 2
cd /home/diego/Escritorio/smolvlm-realtime-webcam
python3 -m http.server 3000
```

Abrir en browser: `http://localhost:3000`

---

## Android / S25 Ultra (Termux)

> Corre todo standalone en el celular — modelo + cámara + TTS sin PC.

### Requisitos
- Samsung Galaxy S25 Ultra (Snapdragon 8 Elite, 12GB RAM)
- Termux (instalar desde **F-Droid**, no Play Store)
- Termux:API (Play Store o F-Droid)

### Instalación única (ya hecha en el S25)

```bash
# 1. Actualizar paquetes
pkg update && pkg upgrade -y

# 2. Instalar dependencias
pkg install clang cmake ninja git python termux-api shaderc vulkan-tools vulkan-loader-android vulkan-headers tmux openssh -y

# 3. Instalar Python deps
pip install requests

# 4. Compilar llama.cpp SIN Vulkan (Vulkan causa crashes en Adreno)
git clone --depth=1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j$(nproc) --target llama-server

# 5. Clonar repo PathsAI
git clone https://github.com/ngxson/smolvlm-realtime-webcam ~/Qwen3-VL-2B-llama.cpp-android
cp ~/Qwen3-VL-2B-llama.cpp-android/index.html ~/pathsai/index.html
```

> **Nota**: Compilar con `-DGGML_VULKAN=ON` causa `vk::DeviceLostError` al procesar imágenes en el Adreno 830. Usar CPU es estable y suficientemente rápido en el Snapdragon 8 Elite.

### Correr todo (cada vez) — Opción A: Chrome + TTS web

```bash
# Sesión tmux para mantener todo corriendo con pantalla apagada
tmux new-session -s pathsai

# Ventana 1: modelo
~/llama.cpp/build/bin/llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --port 8080 --ctx-size 2048 --parallel 1 --threads 8

# Ctrl+B C → nueva ventana
# Ventana 2: servidor web (esperar que ventana 1 diga "listening")
cd ~/pathsai && python -m http.server 3000
```

Abrir Chrome en el S25: `http://localhost:3000`
Activar TTS → presionar Start.

### Correr todo (cada vez) — Opción B: Termux TTS sin Chrome

```bash
tmux new-session -s pathsai

# Ventana 1: modelo
~/llama.cpp/build/bin/llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --port 8080 --ctx-size 2048 --parallel 1 --threads 8

# Ctrl+B C → nueva ventana
# Ventana 2: asistente con voz
termux-wake-lock && python ~/paths_ai_termux.py --url http://localhost:8080 --camera 0 --interval 3
```

Funciona con pantalla apagada. `Ctrl+C` para detener.

### Comandos tmux útiles

| Acción | Comando |
|--------|---------|
| Salir sin matar procesos | `Ctrl+B` luego `D` |
| Volver a la sesión | `tmux attach -t pathsai` |
| Nueva ventana | `Ctrl+B C` |
| Cambiar ventana | `Ctrl+B 0` / `Ctrl+B 1` |
| Matar sesión | `tmux kill-session -t pathsai` |

---

## Encender Otro Día

### En PC (Fedora + RTX 4070)

```bash
# Terminal 1: modelo
llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf --port 8080 -ngl 99 --ctx-size 4096 --parallel 1

# Terminal 2: frontend web
cd /home/diego/Escritorio/smolvlm-realtime-webcam && python3 -m http.server 3000
```

Abrir: `http://localhost:3000`

### En S25 Ultra (Termux standalone)

```bash
tmux new-session -s pathsai
~/llama.cpp/build/bin/llama-server --hf-repo Qwen/Qwen3-VL-2B-Instruct-GGUF --hf-file Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --port 8080 --ctx-size 2048 --parallel 1 --threads 8
# Ctrl+B C
cd ~/pathsai && python -m http.server 3000
```

Abrir Chrome: `http://localhost:3000`

### En celular conectado a PC (WiFi)

```bash
# Ver IP de la PC
ip addr show | grep "inet " | grep -v 127.0.0.1

# En el celular abrir Chrome:
# http://IP-DE-PC:3000
```

---

## Estructura del Repositorio

```
smolvlm-realtime-webcam/
├── index.html              # Frontend web: cámara + TTS + prompt PathsAI
├── paths_ai_mvp.py         # Script Python PC: OpenCV + pyttsx3 + logs JSON
├── paths_ai_termux.py      # Script Android: termux-camera + termux-tts-speak
├── pyproject.toml          # Dependencias Python (uv)
├── uv.lock                 # Lock file reproducible
├── DEPLOYMENT.md           # Esta guía
├── demo.png                # Captura del demo
└── LICENSE
```

---

## Rendimiento Observado

### PC — RTX 4070 Laptop + CUDA

| Métrica | Valor |
|---------|-------|
| Procesamiento imagen | ~61 ms |
| Generación respuesta | ~80 ms |
| **Total por frame** | **~140 ms** |

### S25 Ultra — Snapdragon 8 Elite (CPU, sin Vulkan)

| Métrica | Valor estimado |
|---------|----------------|
| Procesamiento imagen | ~500–1000 ms |
| Generación respuesta | ~2–4s |
| **Total por frame** | **~3–5s** |

---

## Prompt PathsAI

El prompt está diseñado para navegación de personas invidentes. Siempre responde con 3 partes:

1. **Contexto** — qué hay adelante (entorno general)
2. **Obstáculos** — qué, dónde y a qué distancia
3. **Instrucción** — acción directa (gira, detente, continúa)

Ejemplos de respuesta:
- *"Estás en un pasillo. Persona a la derecha, cerca. Desvíate a la izquierda."*
- *"Hay una sala con muebles. Silla al centro, muy cerca. Detente y rodea por la derecha."*
- *"Estás frente a una escalera. Escalones hacia abajo, muy cerca. Detente, baja con cuidado."*
- *"Hay una calle con personas. Grupo al frente, a distancia. Cambia de ruta, gira a la derecha."*
- *"Camino despejado, continúa adelante."*

---

> **Proyecto**: PathsAI — Huawei ICT Competition 2026, Innovation Track
> **Stack**: Qwen3-VL-2B (GGUF) + llama.cpp + Web Speech API / Termux TTS
