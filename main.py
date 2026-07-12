from __future__ import annotations

import base64
import io
import math
import os
import statistics
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

import miniaudio
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Audio Stats API", version="1.0.0")
WHISPER_MODEL = None
WHISPER_MODULE = None
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")


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


def _get_whisper_model():
	global WHISPER_MODEL, WHISPER_MODULE
	if WHISPER_MODULE is None:
		try:
			import whisper as whisper_module
		except Exception as exc:
			raise HTTPException(status_code=503, detail="Transcription dependency is unavailable") from exc
		WHISPER_MODULE = whisper_module

	if WHISPER_MODEL is None:
		WHISPER_MODEL = WHISPER_MODULE.load_model(WHISPER_MODEL_NAME)
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
		result = model.transcribe(str(temp_path), language="ko", fp16=False)
		text = str(result.get("text", "")).strip()
	except Exception as exc:
		raise HTTPException(status_code=400, detail="Unable to transcribe audio") from exc
	finally:
		temp_path.unlink(missing_ok=True)

	if not text:
		raise HTTPException(status_code=400, detail="Unable to transcribe audio")

	return text


def _build_response(df: pd.DataFrame, audio_id: str, sample_rate: int, sample_count: int) -> dict[str, Any]:
	response = {
		key: value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value
		for key, value in RESPONSE_TEMPLATE.items()
	}
	response["rows"] = int(df.shape[0])
	response["columns"] = [str(column) for column in df.columns.tolist()]

	summary_targets = {
		"mean": df.mean(numeric_only=True),
		"std": df.std(numeric_only=True, ddof=0),
		"variance": df.var(numeric_only=True, ddof=0),
		"min": df.min(numeric_only=True),
		"max": df.max(numeric_only=True),
		"median": df.median(numeric_only=True),
	}

	for key, series in summary_targets.items():
		response[key] = {str(column): float(value) for column, value in series.items()}

	mode_values: dict[str, Any] = {}
	allowed_values: dict[str, list[Any]] = {}
	range_values: dict[str, float] = {}
	for column in df.columns:
		column_series = df[column]
		if pd.api.types.is_numeric_dtype(column_series):
			values = [float(value) for value in column_series.dropna().tolist()]
			if values:
				try:
					mode_value = statistics.mode(values)
				except statistics.StatisticsError:
					mode_value = values[0]
				mode_values[str(column)] = float(mode_value)
				range_values[str(column)] = float(max(values) - min(values))
		else:
			unique_values = [value for value in column_series.dropna().astype(str).tolist() if value]
			if unique_values:
				mode_values[str(column)] = unique_values[0]
				allowed_values[str(column)] = unique_values

	response["mode"] = mode_values
	response["range"] = range_values
	response["allowed_values"] = allowed_values
	response["value_range"] = {
		"audio_id": audio_id,
		"sample_rate": int(sample_rate),
		"sample_count": int(sample_count),
	}

	if df.shape[0] > 1 and df.select_dtypes(include="number").shape[1] > 1:
		corr = df.corr(numeric_only=True).fillna(0.0)
		response["correlation"] = corr.round(6).values.tolist()
	else:
		response["correlation"] = []

	return response


@app.post("/analyze")
def analyze_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	transcript = _transcribe_audio(audio_bytes)
	df = pd.DataFrame([{"transcript": transcript}])
	response = _build_response(df, payload.audio_id, 0, 1)
	response["columns"] = [transcript]
	return response


@app.get("/")
def root() -> dict[str, str]:
	return {"status": "ok", "endpoint": "/analyze"}


@app.post("/preview")
def preview_audio(payload: AudioRequest) -> dict[str, Any]:
	audio_bytes = _decode_audio_bytes(payload.audio_base64)
	transcript = _transcribe_audio(audio_bytes)
	df = pd.DataFrame([{"transcript": transcript}])
	response = _build_response(df, payload.audio_id, 0, 1)
	response["columns"] = [transcript]
	return response
