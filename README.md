# Fish Speech API

> **Built on top of [Fish Speech](https://github.com/fishaudio/fish-speech)** by [Fish Audio / 39 AI, Inc](https://fish.audio/). This is an independent project — not affiliated with or endorsed by Fish Audio. The model and inference code are licensed under the **[Fish Audio Research License](LICENSE)** — research and non-commercial use only. For commercial use, see the [official repo](https://github.com/fishaudio/fish-speech).

A self-hosted, GPU-accelerated Text-to-Speech REST API powered by [Fish Speech](https://github.com/fishaudio/fish-speech) and the **OpenAudio S1-mini** model (0.5B parameters).

Packaged as a single Docker image with a lightweight FastAPI server. Supports zero-shot voice cloning, emotion/tone control, and simple integration via JSON or multipart endpoints.

**Why this exists:** The official Fish Speech Docker images (`fishaudio/fish-speech:latest`) crashed on startup for me — the server fails to load the `dual_ar` model architecture (`UnboundLocalError: cannot access local variable 'tokenizer'`), and when the checkpoint volume is empty, it crashes with `FileNotFoundError` on `config.json` with no auto-download mechanism. See [fishaudio/fish-speech#6](https://github.com/fishaudio/fish-speech/issues/6) and similar issues. I couldn't fix it, so I built this minimal wrapper instead — automatic model download via `HF_TOKEN`, a simple REST API, and easy integration into any existing Docker project as a git submodule (just add the submodule, drop the service in your `docker-compose.yml`, set `HF_TOKEN` in `.env`, and you're done).

## Supported Models

| Model | Parameters | Status |
|-------|-----------|--------|
| [OpenAudio S1-mini](https://huggingface.co/fishaudio/s1-mini) | 0.5B | Supported |
| [OpenAudio S2-pro](https://huggingface.co/fishaudio/s2-pro) | 5B | Coming soon |

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Model Setup](#model-setup)
- [Running the Container](#running-the-container)
- [API Reference](#api-reference)
- [Auto-chunking TTS](#post-v1ttsauto)
- [Streaming mode](#streaming-mode)
- [Voice Cloning](#voice-cloning)
- [Emotion and Tone Markers](#emotion-and-tone-markers)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- GPU-accelerated inference (CUDA)
- Zero-shot voice cloning from a short audio sample
- 51 emotion tags + tone/sound effect markers
- Two API endpoints: full-featured JSON and simple multipart form
- Automatic model download when `HF_TOKEN` is provided
- Docker-native with volume-based model persistence
- Automatic chunking for long texts (`/v1/tts/auto` — split + concatenate)
- Optional streaming mode on `/v1/tts/auto` for lower first-byte latency
- **Coming soon:** S2-pro model support

## Requirements

| Requirement | Details |
|---|---|
| Docker | 20.10+ with BuildKit |
| NVIDIA GPU | Compute Capability 6.0+ (Pascal or newer) |
| nvidia-container-toolkit | [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| VRAM | ~5 GB minimum (tested on RTX 2060 6 GB) |
| Disk | ~3 GB for the Docker image + ~2 GB for the model |

## Quick Start

If you have an NVIDIA GPU, Docker, and a HuggingFace token:

```bash
# Build
docker build -t fish-speech-api .

# Configure your HuggingFace token
cp .env.example .env
# Edit .env and set your HF_TOKEN (Read access, get one at https://huggingface.co/settings/tokens)

# Run (model auto-downloads on first start)
docker run --gpus all -p 8080:8080 --env-file .env -v fish-speech-checkpoints:/app/checkpoints fish-speech-api

# Test
curl http://localhost:8080/v1/health
# {"status": "ok", "device": "cuda"}

# Generate speech
curl -X POST http://localhost:8080/v1/tts/test \
  -F "text=Hello, this is Fish Speech running in Docker!" \
  --output hello.wav
```

The model is downloaded into the Docker volume on the first run and persisted for subsequent starts.

## Building the Image

```bash
git clone git@github.com:laurentf/fish-speech-api.git
cd fish-speech-api
docker build -t fish-speech-api .
```

The build takes a few minutes. The resulting image is based on `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` and includes all Python dependencies pre-installed.

## Model Setup

The API requires the **OpenAudio S1-mini** model checkpoint. There are two ways to set it up:

### Option 1: Automatic download (recommended)

Set `HF_TOKEN` in a `.env` file. The entrypoint script will automatically download the model from HuggingFace on the first start if it is not already present in the checkpoints volume.

```bash
# Create your .env from the example
cp .env.example .env
# Edit .env and set your HF_TOKEN

docker run --gpus all -p 8080:8080 --env-file .env -v fish-speech-checkpoints:/app/checkpoints fish-speech-api
```

To get a HuggingFace token:
1. Create an account on [huggingface.co](https://huggingface.co)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a token with **Read** access

The download happens once. On subsequent starts, the model is loaded directly from the volume.

### Option 2: Manual download

Download the model on your host machine and mount it into the container.

```bash
# Install the HuggingFace CLI
pip install huggingface_hub

# Login (if the model requires authentication)
huggingface-cli login

# Download the model
huggingface-cli download fishaudio/openaudio-s1-mini \
  --local-dir ./openaudio-s1-mini

# Run with a bind mount
docker run --gpus all -p 8080:8080 -v $(pwd)/openaudio-s1-mini:/app/checkpoints/openaudio-s1-mini fish-speech-api
```

Alternatively, using `uvx`:

```bash
uvx --from huggingface_hub huggingface-cli download fishaudio/openaudio-s1-mini \
  --local-dir ./openaudio-s1-mini
```

### Volume structure

The container expects the model files at `/app/checkpoints/openaudio-s1-mini/`. The directory must contain (among others) the `codec.pth` file. The entrypoint checks for this file to determine if the model is present.

```
/app/checkpoints/
└── openaudio-s1-mini/
    ├── codec.pth
    ├── model.pth
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## Running the Container

### Basic usage

```bash
docker run --gpus all -p 8080:8080 -v fish-speech-checkpoints:/app/checkpoints fish-speech-api
```

### With automatic model download

```bash
docker run --gpus all -p 8080:8080 --env-file .env -v fish-speech-checkpoints:/app/checkpoints fish-speech-api
```

### With docker-compose

```yaml
services:
  fish-speech:
    build: .
    image: fish-speech-api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8080:8080"
    volumes:
      - fish-speech-checkpoints:/app/checkpoints
    environment:
      - HF_TOKEN=${HF_TOKEN}  # set in .env or shell

volumes:
  fish-speech-checkpoints:
```

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f fish-speech
```

### Using from another container

If you run Fish Speech alongside other services in a `docker-compose` setup, other containers can reach it via the service name as hostname. No need to expose port 8080 to the host.

```yaml
services:
  my-app:
    image: my-app:latest
    environment:
      - FISH_SPEECH_URL=http://fish-speech:8080

  fish-speech:
    image: fish-speech-api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - fish-speech-checkpoints:/app/checkpoints
    env_file: .env  # must contain HF_TOKEN

volumes:
  fish-speech-checkpoints:
```

From `my-app`, call the API at `http://fish-speech:8080/v1/tts/auto` — Docker's internal DNS resolves the service name automatically.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | No | HuggingFace API token (Read access). If set, the model is auto-downloaded on first start. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |

## API Reference

The server exposes four endpoints on port **8080**.

### GET /v1/health

Health check. Returns device info.

```bash
curl http://localhost:8080/v1/health
```

```json
{"status": "ok", "device": "cuda"}
```

> **Tip:** If you don't want to think about text length, always use `/v1/tts/auto` — it handles any length and will generate as many chunks as needed (even just one for short texts).

### POST /v1/tts/simple

Single-chunk TTS endpoint. Accepts JSON with base64 references. Returns audio bytes. For short texts only — no automatic chunking.

**Content-Type:** `application/json`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | *required* | Text to synthesize. Supports [emotion markers](#emotion-and-tone-markers). |
| `references` | array | `[]` | Voice cloning references: `[{"audio": "<base64>", "text": "<transcript>"}]` |
| `reference_id` | string | `null` | Pre-registered reference voice ID |
| `chunk_length` | int | `200` | Chunk length for processing (100-300) |
| `max_new_tokens` | int | `1024` | Maximum generated tokens (100-4096) |
| `top_p` | float | `0.7` | Nucleus sampling threshold (0.1-1.0) |
| `repetition_penalty` | float | `1.2` | Repetition penalty (0.9-2.0) |
| `temperature` | float | `0.7` | Sampling temperature (0.1-1.0) |
| `seed` | int | `null` | Random seed for reproducibility |
| `normalize` | bool | `true` | Normalize text (expand numbers, dates, etc.) |
| `format` | string | `"wav"` | Output format: `wav` or `pcm` |
| `use_memory_cache` | string | `"off"` | Enable memory cache: `on` or `off` |

**Example with curl:**

```bash
curl -X POST http://localhost:8080/v1/tts/simple \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "temperature": 0.8}' \
  --output output.wav
```

**Example with Python:**

```python
import base64
import requests

# Without voice cloning
response = requests.post("http://localhost:8080/v1/tts/simple", json={
    "text": "(excited)Hello everyone! This is Fish Speech!",
    "temperature": 0.8,
})
with open("output.wav", "wb") as f:
    f.write(response.content)

# With voice cloning
with open("my_voice.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8080/v1/tts/simple", json={
    "text": "This should sound like the reference voice.",
    "references": [{
        "audio": audio_b64,
        "text": "Exact transcript of the reference audio."
    }],
})
with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

### POST /v1/tts/test

Test TTS endpoint using multipart form data. Designed for quick testing via Swagger — upload a reference audio file directly without base64 encoding. Uses default generation parameters.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | Text to synthesize |
| `reference_audio` | file | No | Audio file for voice cloning |
| `reference_text` | string | No | Transcript of the reference audio |

**Example with curl:**

```bash
# Simple text-to-speech
curl -X POST http://localhost:8080/v1/tts/test \
  -F "text=Hello world" \
  --output output.wav

# With voice cloning
curl -X POST http://localhost:8080/v1/tts/test \
  -F "text=Hello world" \
  -F "reference_audio=@my_voice.wav" \
  -F "reference_text=Transcript of my voice sample" \
  --output output.wav
```

**Example with Python:**

```python
import requests

# Simple
response = requests.post("http://localhost:8080/v1/tts/test",
    data={"text": "Hello from Fish Speech!"})
with open("output.wav", "wb") as f:
    f.write(response.content)

# With voice cloning
with open("my_voice.wav", "rb") as f:
    response = requests.post("http://localhost:8080/v1/tts/test",
        data={
            "text": "(soft tone)Hello, this is a cloned voice.",
            "reference_text": "Transcript of the reference audio."
        },
        files={"reference_audio": ("voice.wav", f, "audio/wav")},
    )
with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

### POST /v1/tts/auto

Long-text TTS with automatic chunking. Splits the input text into sentences, generates audio for each chunk, and concatenates them into a single WAV. Use this when your text exceeds the model's token limit.

**Content-Type:** `application/json`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | *required* | Long text to synthesize |
| `references` | array | `[]` | Voice cloning references (same as `/v1/tts/simple`) |
| `reference_id` | string | `null` | Pre-registered reference voice ID |
| `chunk_length_bytes` | int | `200` | Max chunk size in UTF-8 bytes for text splitting (50-500) |
| `chunk_length` | int | `200` | Model chunk length for inference (100-300) |
| `max_new_tokens` | int | `1024` | Maximum generated tokens per chunk (100-4096) |
| `top_p` | float | `0.7` | Nucleus sampling threshold (0.1-1.0) |
| `repetition_penalty` | float | `1.2` | Repetition penalty (0.9-2.0) |
| `temperature` | float | `0.7` | Sampling temperature (0.1-1.0) |
| `seed` | int | `null` | Random seed for reproducibility |
| `normalize` | bool | `true` | Normalize text |
| `streaming` | bool | `false` | Stream audio chunks as they are generated (see [Streaming mode](#streaming-mode)) |
| `use_memory_cache` | string | `"off"` | Memory cache (`on`, `off`) |

The text is split at sentence boundaries (`.`, `!`, `?`), then at commas, then at spaces if chunks are still too long. Voice cloning references are applied to every chunk for consistent voice.

**Example with curl:**

```bash
curl -X POST http://localhost:8080/v1/tts/auto \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour à tous et bienvenue dans cette émission spéciale. Aujourd'\''hui nous allons explorer comment les modèles de langage ont révolutionné la façon dont les machines parlent. Il y a dix ans, les voix synthétiques sonnaient comme des robots. Aujourd'\''hui, on obtient des résultats bluffants de réalisme. Merci d'\''avoir écouté!"}' \
  --output long_output.wav
```

**Example with Python:**

```python
import requests

response = requests.post("http://localhost:8080/v1/tts/auto", json={
    "text": "Your very long text here. It will be automatically split into chunks. "
            "Each chunk is synthesized separately and concatenated into a single WAV. "
            "Voice cloning references are applied to every chunk for consistency.",
    "chunk_length_bytes": 200,
})
with open("long_output.wav", "wb") as f:
    f.write(response.content)
```

**Performance (RTX 2060 6GB):** ~5 tokens/sec, ~25-30s per chunk. A 6-chunk text (~850 bytes) takes ~3 minutes total.

### Streaming mode

By default, `/v1/tts/auto` waits for all chunks to be generated and returns a single concatenated WAV file. With `streaming: true`, audio segments are sent to the client as soon as they are decoded, reducing the time to first audio byte.

**How it works:** The response is a chunked HTTP stream (`Transfer-Encoding: chunked`). The first bytes are a WAV header, followed by raw int16 PCM segments as they come out of the decoder. Each text chunk (sentence) is processed with the engine's native streaming, so you receive audio segments within a chunk too — not just one blob per sentence.

**Important: `streaming` defaults to `false` for a reason.** Streaming only works smoothly if the GPU generates audio faster than real-time. If the next segment takes longer to generate than the previous one takes to play, the client will experience gaps/silence between chunks. This depends entirely on your hardware:

| GPU | Real-time factor | Streaming viable? |
|---|---|---|
| RTX 4090 / A100 | ~0.2-0.3x | Yes |
| RTX 3090 | ~0.3-0.5x | Yes |
| RTX 3070 / 3080 | ~0.5-0.8x | Borderline |
| RTX 2060 / 3060 | ~0.8-1.5x | No — use default mode |
| CPU | >>1x | No |

If you're unsure, keep the default (`streaming: false`). You'll wait longer for the first byte, but the resulting audio is guaranteed to be continuous with no gaps.

**Example — streaming with Python:**

```python
import requests

response = requests.post("http://localhost:8080/v1/tts/auto", json={
    "text": "A long text that will be split into chunks automatically.",
    "streaming": True,
}, stream=True)

# Write chunks to file as they arrive
with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=4096):
        f.write(chunk)
```

**Example — streaming to an audio player (low-latency pipeline):**

```python
import requests

response = requests.post("http://localhost:8080/v1/tts/auto", json={
    "text": "First sentence arrives fast. Then the rest follows chunk by chunk.",
    "streaming": True,
}, stream=True)

for chunk in response.iter_content(chunk_size=4096):
    audio_player.feed(chunk)  # play immediately as chunks arrive
```

## Voice Cloning

Fish Speech supports **zero-shot voice cloning**: provide a short audio sample of any voice, and the model will generate speech that sounds like it. No fine-tuning needed.

### How it works

You provide two things:
1. **A reference audio clip** of the voice you want to clone
2. **An exact transcript** of what is said in that audio clip

The model uses both to learn the voice characteristics (timbre, pitch, rhythm, accent) and applies them to your target text.

### Preparing a good reference

| Aspect | Recommendation |
|---|---|
| **Duration** | 5-10 seconds is the sweet spot. Under 3s gives poor results. Over 15s may cause errors or slow inference. |
| **Speaking style** | Use a **natural, conversational tone**. Avoid whispering, shouting, or exaggerated emotions in the reference -- the model will pick those up and reproduce them. |
| **Audio quality** | Clean recording, no background music, no reverb, no other speakers. A smartphone voice memo in a quiet room works fine. |
| **Content** | The reference should contain varied intonation (not a monotone reading). A sentence with commas, questions, or natural pauses works best. |
| **Format** | WAV, 24000 Hz, mono. Other formats (mp3, m4a, flac) work but are converted internally. |

**Convert any audio file to the optimal format:**

```bash
ffmpeg -i input.m4a -ar 24000 -ac 1 reference.wav
```

### The transcript matters

The `reference_text` (or `text` field in the reference object) must be the **exact, word-for-word transcript** of what is spoken in the reference audio. This is not optional -- the model uses it to align the voice characteristics.

- Wrong transcript = distorted or garbled output
- Missing words or extra words = unpredictable quality
- Punctuation matters: include natural pauses (commas, periods) as spoken

If you don't know the exact transcript, use a transcription tool (Whisper, etc.) to generate one, then verify it manually.

### Sending references via the API

**Option A: File upload** (`/v1/tts/test` -- multipart form)

The simplest approach for testing voice cloning through Swagger without writing code. Send the audio file directly:

```bash
curl -X POST http://localhost:8080/v1/tts/test \
  -F "text=This will sound like the reference voice." \
  -F "reference_audio=@reference.wav" \
  -F "reference_text=The exact words spoken in the reference audio." \
  --output output.wav
```

```python
with open("reference.wav", "rb") as f:
    response = requests.post("http://localhost:8080/v1/tts/test",
        data={
            "text": "This will sound like the reference voice.",
            "reference_text": "The exact words spoken in the reference audio."
        },
        files={"reference_audio": ("ref.wav", f, "audio/wav")},
    )
```

**Option B: Base64-encoded** (`/v1/tts/auto` or `/v1/tts/simple` -- JSON)

For programmatic use. Encode the reference audio as base64. Prefer `/v1/tts/auto` — it handles both short and long texts (automatic chunking when needed), and is the recommended endpoint for integration:

```python
import base64, requests

with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8080/v1/tts/auto", json={
    "text": "This will sound like the reference voice, even for very long texts.",
    "references": [{
        "audio": audio_b64,
        "text": "The exact words spoken in the reference audio."
    }],
})
```

You can pass multiple references in the array to give the model more examples of the same voice. Each reference needs its own audio + transcript pair.

> **Note:** `/v1/tts/simple` works the same way but without chunking — use it only for short texts.

### Troubleshooting

| Problem | Likely cause |
|---|---|
| Output sounds nothing like the reference | Transcript doesn't match the audio, or audio is too short |
| Garbled / distorted output | Transcript has extra or missing words |
| Model returns an error | Reference audio is too long (try trimming to 10s) |
| Voice sounds "flat" | Reference audio is monotone -- re-record with natural intonation |
| Accent is wrong | The model picks up accent from the reference -- use a reference in the target accent |

## Emotion and Tone Markers

The S1-mini model supports inline emotion and tone control using `(tag)` markers in the text.

### Emotions (51)

```
(excited) (amused) (joyful) (happy) (sad) (angry) (furious) (surprised)
(scared) (anxious) (nervous) (worried) (confused) (curious) (interested)
(confident) (proud) (grateful) (relaxed) (satisfied) (delighted) (moved)
(empathetic) (sincere) (comforting) (embarrassed) (awkward) (guilty)
(frustrated) (upset) (unhappy) (depressed) (disappointed) (disgusted)
(disdainful) (scornful) (sneering) (sarcastic) (impatient) (indifferent)
(reluctant) (hesitating) (yielding) (painful) (panicked) (hysterical)
(serious) (disapproving) (negative) (denying) (astonished) (keen)
```

### Tones

```
(soft tone) (whispering) (shouting) (screaming) (in a hurry tone)
```

### Sound effects

```
(laughing) (chuckling) (sighing) (sobbing) (crying loudly)
(panting) (groaning) (crowd laughing) (background laughter)
```

### Usage example

```
(excited)Hey everyone!(laughing) Welcome to the show!(soft tone)Now let me tell you something special...
```

## Performance

Tested on NVIDIA RTX 2060 (6 GB VRAM):

| Metric | Value |
|---|---|
| VRAM usage | ~4.7 GB |
| Inference speed | ~5 tokens/sec |
| Memory bandwidth | ~4 GB/s |
| Latency (short sentence) | ~5-6 seconds |

## Project Structure

```
fish-speech-api/
├── server.py            # FastAPI server with /v1/tts/test, /v1/tts/simple, /v1/tts/auto
├── Dockerfile           # Docker image definition
├── entrypoint.sh        # Entrypoint: auto-downloads model if HF_TOKEN is set
├── .env.example         # Environment variables template (HF_TOKEN)
├── requirements.txt     # Python dependencies
├── fish_speech/         # Fish Speech inference library (from upstream)
└── tools/               # Utility scripts (from upstream)
```

## License

This model is licensed under the [Fish Audio Research License](LICENSE), Copyright © 39 AI, INC. All Rights Reserved.
