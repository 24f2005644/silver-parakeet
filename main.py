from __future__ import annotations

import base64
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Audio Stats API", version="1.0.0")
WHISPER_MODEL = None


class AudioRequest(BaseModel):
	audio_id: str = Field(..., description="Audio sample identifier")
	audio_base64: str = Field(..., description="Base64-encoded audio bytes")


RESPONSE_TEMPLATE: dict[str, Any] = {
	"rows": 0,
	"columns": [],
	"mean": {},
	"std": {},
	"variance": {},
	"min": {},
	"max": {},
	"median": {},
	"mode": {},
	"range": {},
	"allowed_values": {},
	"value_range": {},
	"correlation": [],
}


def _decode_audio_bytes(audio_base64: str) -> bytes:
	try:
		return base64.b64decode(audio_base64, validate=True)
	except Exception as exc:
		raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc


def _guess_audio_suffix(audio_bytes: bytes) -> str:
	if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
		return ".wav"
	if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
		return ".mp3"
	if audio_bytes.startswith(b"fLaC"):
		return ".flac"
	if audio_bytes.startswith(b"OggS"):
		return ".ogg"
	return ".audio"


def _get_whisper_model() -> WhisperModel:
	global WHISPER_MODEL
	if WHISPER_MODEL is None:
		model_name = os.getenv("WHISPER_MODEL_NAME", os.getenv("WHISPER_MODEL", "tiny"))
		compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
		cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "1"))
		WHISPER_MODEL = WhisperModel(
			model_name,
			device="cpu",
			compute_type=compute_type,
			cpu_threads=cpu_threads,
			num_workers=1,
		)
	return WHISPER_MODEL


def _transcribe_audio(audio_bytes: bytes) -> str:
	with tempfile.NamedTemporaryFile(delete=False, suffix=_guess_audio_suffix(audio_bytes)) as temp_file:
		temp_file.write(audio_bytes)
		temp_path = Path(temp_file.name)

	try:
		model = _get_whisper_model()
		segments, _info = model.transcribe(str(temp_path), language="ko", beam_size=1, vad_filter=False)
		text = "".join(segment.text for segment in segments).strip()
	except Exception as exc:
		raise HTTPException(status_code=400, detail="Unable to transcribe audio") from exc
	finally:
		temp_path.unlink(missing_ok=True)

	if not text:
		raise HTTPException(status_code=400, detail="Unable to transcribe audio")

	return text


def _build_response(audio_id: str, transcript: str) -> dict[str, Any]:
	response = deepcopy(RESPONSE_TEMPLATE)
	response["rows"] = 1
	response["columns"] = [transcript]
	response["mode"] = {"transcript": transcript}
	response["allowed_values"] = {"transcript": [transcript]}
	response["value_range"] = {"audio_id": audio_id}
	return response


@app.post("/analyze")
def analyze_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	transcript = _transcribe_audio(audio_bytes)
	return _build_response(payload.audio_id, transcript)


@app.get("/")
def root() -> dict[str, str]:
	return {"status": "ok", "endpoint": "/analyze"}


@app.post("/preview")
def preview_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	transcript = _transcribe_audio(audio_bytes)
	return _build_response(payload.audio_id, transcript)
