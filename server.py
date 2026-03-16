import os
import io
import wave
import base64
import torch
import numpy as np
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup
os.makedirs("checkpoints", exist_ok=True)
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from loguru import logger
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.text.spliter import split_text

app = FastAPI(title="Fish Speech API", version="1.0.0")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# Global engine
engine = None


class ReferenceAudio(BaseModel):
    """Reference audio for voice cloning"""
    audio: str  # Base64 encoded audio
    text: str   # Transcript of the reference audio


class TTSRequest(BaseModel):
    """TTS request with all options"""
    text: str

    # Voice cloning references
    references: list[ReferenceAudio] = []
    reference_id: str | None = None

    # Generation parameters
    chunk_length: int = Field(default=200, ge=100, le=300)
    max_new_tokens: int = Field(default=1024, ge=100, le=4096)
    top_p: float = Field(default=0.7, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=0.9, le=2.0)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)

    # Other options
    seed: int | None = None
    normalize: bool = True
    format: Literal["wav", "pcm", "mp3"] = "wav"
    use_memory_cache: Literal["on", "off"] = "off"


@app.on_event("startup")
async def startup():
    global engine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    checkpoint_path = Path("checkpoints/openaudio-s1-mini")

    logger.info("Loading LLAMA model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(checkpoint_path),
        device=device,
        precision=torch.float16,
        compile=False,
    )

    logger.info("Loading decoder model...")
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(checkpoint_path / "codec.pth"),
        device=device,
    )

    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=torch.float16,
        compile=False,
    )

    logger.info("Fish Speech API ready!")


@app.get("/v1/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/v1/tts/simple", responses={200: {"content": {"audio/wav": {}}}})
async def tts_simple_json(request: TTSRequest):
    """
    Single-chunk TTS endpoint (JSON). For short texts with full control over parameters.
    For voice cloning, provide references with base64-encoded audio and transcript.
    For any text length, prefer /v1/tts/auto which handles chunking automatically.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        # Convert references to ServeReferenceAudio
        serve_references = []
        for ref in request.references:
            try:
                audio_bytes = base64.b64decode(ref.audio)
                serve_references.append(ServeReferenceAudio(
                    audio=audio_bytes,
                    text=ref.text
                ))
            except Exception as e:
                logger.warning(f"Failed to decode reference audio: {e}")

        tts_request = ServeTTSRequest(
            text=request.text,
            references=serve_references,
            reference_id=request.reference_id,
            chunk_length=request.chunk_length,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            seed=request.seed,
            normalize=request.normalize,
            format=request.format,
            use_memory_cache=request.use_memory_cache,
        )

        # Generate audio
        result = None
        for chunk in engine.inference(tts_request):
            result = chunk

        if result is None:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Check for errors
        if hasattr(result, 'code') and result.code == "error":
            raise HTTPException(status_code=500, detail=str(result.error))

        # InferenceResult has .audio = (sample_rate, audio_array)
        if hasattr(result, 'audio') and result.audio is not None:
            sample_rate, audio = result.audio
            logger.info(f"Generated audio: sample_rate={sample_rate}, duration={len(audio)/sample_rate:.2f}s")
        else:
            raise HTTPException(status_code=500, detail="No audio in result")

        # Flatten if needed
        if len(audio.shape) > 1:
            audio = audio.flatten()

        # Convert to requested format
        if request.format == "wav":
            audio_data = (audio * 32767).astype(np.int16)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
            )

        elif request.format == "pcm":
            audio_data = (audio * 32767).astype(np.int16)
            return Response(
                content=audio_data.tobytes(),
                media_type="audio/pcm",
                headers={"Content-Disposition": "attachment; filename=tts_output.pcm"},
            )

        else:  # mp3 - would need additional library
            raise HTTPException(status_code=400, detail="MP3 format not yet supported")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/test", responses={200: {"content": {"audio/wav": {}}}})
async def tts_test(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_text: str = Form(None),
):
    """
    Test TTS endpoint with multipart form data.
    Designed for quick testing via Swagger — upload a reference audio file directly without base64 encoding.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        serve_references = []

        # Handle uploaded reference audio
        if reference_audio and reference_text:
            audio_bytes = await reference_audio.read()
            serve_references.append(ServeReferenceAudio(
                audio=audio_bytes,
                text=reference_text
            ))

        tts_request = ServeTTSRequest(
            text=text,
            references=serve_references,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )

        # Generate audio
        result = None
        for chunk in engine.inference(tts_request):
            result = chunk

        if result is None or not hasattr(result, 'audio') or result.audio is None:
            raise HTTPException(status_code=500, detail="No audio generated")

        sample_rate, audio = result.audio

        if len(audio.shape) > 1:
            audio = audio.flatten()

        audio_data = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_chunk(text: str, serve_references: list, params: dict) -> tuple[int, np.ndarray]:
    """Generate audio for a single text chunk. Returns (sample_rate, audio_array)."""
    tts_request = ServeTTSRequest(
        text=text,
        references=serve_references,
        **params,
    )

    result = None
    for chunk in engine.inference(tts_request):
        result = chunk

    if result is None or not hasattr(result, 'audio') or result.audio is None:
        raise HTTPException(status_code=500, detail=f"No audio generated for chunk: {text[:50]}...")

    if hasattr(result, 'code') and result.code == "error":
        raise HTTPException(status_code=500, detail=str(result.error))

    sample_rate, audio = result.audio
    if len(audio.shape) > 1:
        audio = audio.flatten()

    return sample_rate, audio


def _audio_to_wav(sample_rate: int, audio: np.ndarray) -> bytes:
    """Convert audio array to WAV bytes."""
    audio_data = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buffer.getvalue()


class TTSLongRequest(BaseModel):
    """TTS request for long texts with automatic chunking."""
    text: str

    # Voice cloning references
    references: list[ReferenceAudio] = []
    reference_id: str | None = None

    # Chunking
    chunk_length_bytes: int = Field(default=200, ge=50, le=500, description="Max chunk size in UTF-8 bytes for text splitting")

    # Generation parameters
    chunk_length: int = Field(default=200, ge=100, le=300, description="Model chunk length for inference")
    max_new_tokens: int = Field(default=1024, ge=100, le=4096)
    top_p: float = Field(default=0.7, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=0.9, le=2.0)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)

    # Streaming
    streaming: bool = Field(default=False, description="Stream audio chunks as they are generated instead of waiting for full concatenation")

    # Other options
    seed: int | None = None
    normalize: bool = True
    use_memory_cache: Literal["on", "off"] = "off"


def _stream_chunks(chunks: list[str], serve_references: list, params: dict):
    """
    Generator that streams audio chunk by chunk.
    Yields WAV header first, then raw int16 PCM segments for each text chunk.
    """
    header_sent = False

    for i, chunk_text in enumerate(chunks):
        logger.info(f"Streaming chunk {i + 1}/{len(chunks)}: {chunk_text[:50]}...")

        tts_request = ServeTTSRequest(
            text=chunk_text,
            references=serve_references,
            streaming=True,
            **params,
        )

        for result in engine.inference(tts_request):
            if result.code == "error":
                logger.error(f"Streaming TTS error on chunk {i + 1}: {result.error}")
                return
            elif result.code == "header" and not header_sent:
                _, header_bytes = result.audio
                if isinstance(header_bytes, np.ndarray):
                    yield header_bytes.tobytes()
                else:
                    yield bytes(header_bytes)
                header_sent = True
            elif result.code == "segment":
                _, segment = result.audio
                if len(segment.shape) > 1:
                    segment = segment.flatten()
                yield (segment * 32767).astype(np.int16).tobytes()


@app.post("/v1/tts/auto", responses={200: {"content": {"audio/wav": {}}}})
async def tts_auto(request: TTSLongRequest):
    """
    Auto-chunking TTS endpoint (recommended for integration).

    Handles any text length — splits into sentences/segments when needed,
    generates audio for each chunk, and concatenates into a single WAV.
    Works fine with short texts too (single chunk). Use this by default.

    Set streaming=true to receive audio chunks as they are generated (lower latency).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        # Convert references
        serve_references = []
        for ref in request.references:
            try:
                audio_bytes = base64.b64decode(ref.audio)
                serve_references.append(ServeReferenceAudio(
                    audio=audio_bytes,
                    text=ref.text
                ))
            except Exception as e:
                logger.warning(f"Failed to decode reference audio: {e}")

        # Split text into chunks
        chunks = split_text(request.text, request.chunk_length_bytes)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text to synthesize after splitting")

        logger.info(f"Split text into {len(chunks)} chunks: {[c[:30] + '...' if len(c) > 30 else c for c in chunks]}")

        # Generation params (shared across chunks)
        params = dict(
            reference_id=request.reference_id,
            chunk_length=request.chunk_length,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            seed=request.seed,
            normalize=request.normalize,
            format="wav",
            use_memory_cache=request.use_memory_cache,
        )

        # Streaming mode: return audio chunks as they're generated
        if request.streaming:
            return StreamingResponse(
                _stream_chunks(chunks, serve_references, params),
                media_type="audio/wav",
                headers={
                    "Transfer-Encoding": "chunked",
                    "X-Stream-Format": "wav-chunked",
                },
            )

        # Non-streaming mode: generate all chunks and concatenate
        all_audio = []
        sample_rate = None

        for i, chunk_text in enumerate(chunks):
            logger.info(f"Generating chunk {i + 1}/{len(chunks)}: {chunk_text[:50]}...")
            sr, audio = _generate_chunk(chunk_text, serve_references, params)
            sample_rate = sr
            all_audio.append(audio)

        # Concatenate all chunks
        full_audio = np.concatenate(all_audio)
        total_duration = len(full_audio) / sample_rate
        logger.info(f"Generated {len(chunks)} chunks, total duration: {total_duration:.2f}s")

        return Response(
            content=_audio_to_wav(sample_rate, full_audio),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS long error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
