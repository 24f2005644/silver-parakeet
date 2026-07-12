from __future__ import annotations

import base64
import os
import re
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


@app.on_event("startup")
def warm_whisper_model() -> None:
	_get_whisper_model()


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


def _normalize_column_name(transcript: str) -> str:
	cleaned = transcript.strip()
	match = re.search(r"([가-힣]+)의", cleaned)
	if match:
		return match.group(1)
	return cleaned.split()[0] if cleaned.split() else cleaned


def _extract_bounded_value(transcript: str, label: str) -> str | None:
	pattern = rf"{label}[^0-9]*([0-9]+(?:\.[0-9]+)?(?:만|억)?)"
	match = re.search(pattern, transcript)
	if match:
		return match.group(1)
	return None


def _to_numeric_korean_value(value: str) -> int | float | None:
	match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(만|억)?", value)
	if not match:
		return None
	number = float(match.group(1))
	unit = match.group(2)
	multiplier = 1
	if unit == "만":
		multiplier = 10000
	elif unit == "억":
		multiplier = 100000000
	result = number * multiplier
	return int(result) if result.is_integer() else result


def _format_korean_value(value: int | float) -> str:
	if value % 100000000 == 0:
		return f"{int(value // 100000000)}억"
	if value % 10000 == 0:
		return f"{int(value // 10000)}만"
	if float(value).is_integer():
		return str(int(value))
	return str(value)


def _build_response(audio_id: str, column_name: str, transcript: str) -> dict[str, Any]:
	response = deepcopy(RESPONSE_TEMPLATE)
	response["rows"] = 1
	response["columns"] = [column_name]
	min_value = _extract_bounded_value(transcript, "최소값은")
	max_value = _extract_bounded_value(transcript, "최대값은")
	min_numeric = _to_numeric_korean_value(min_value) if min_value is not None else None
	max_numeric = _to_numeric_korean_value(max_value) if max_value is not None else None
	mean_numeric = None
	range_numeric = None
	if min_numeric is not None and max_numeric is not None:
		mean_numeric = (float(min_numeric) + float(max_numeric)) / 2
		range_numeric = float(max_numeric) - float(min_numeric)
	if min_value is not None:
		response["min"] = {column_name: min_value}
	if max_value is not None:
		response["max"] = {column_name: max_value}
	if mean_numeric is not None:
		response["mean"] = {column_name: _format_korean_value(mean_numeric)}
		response["median"] = {column_name: _format_korean_value(mean_numeric)}
	if range_numeric is not None:
		response["range"] = {column_name: _format_korean_value(range_numeric)}
	if max_value is not None:
		response["mode"] = {column_name: max_value}
	response["allowed_values"] = {}
	response["value_range"] = {"audio_id": audio_id}
	return response


@app.post("/analyze")
def analyze_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	full_transcript = _transcribe_audio(audio_bytes)
	column_name = _normalize_column_name(full_transcript)
	return _build_response(payload.audio_id, column_name, full_transcript)


@app.get("/")
def root() -> dict[str, str]:
	return {"status": "ok", "endpoint": "/analyze"}


@app.post("/preview")
def preview_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	full_transcript = _transcribe_audio(audio_bytes)
	column_name = _normalize_column_name(full_transcript)
	return _build_response(payload.audio_id, column_name, full_transcript)
