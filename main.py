from __future__ import annotations

import base64
import io
import math
import statistics
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

app = FastAPI(title="Audio Stats API", version="1.0.0")


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


def _load_wav_samples(audio_bytes: bytes) -> tuple[list[float], int]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw_frames = wav_file.readframes(frame_count)
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Provide WAV audio.") from exc

    if sample_width == 1:
        samples = array("b", raw_frames)
    elif sample_width == 2:
        samples = array("h")
        samples.frombytes(raw_frames)
    elif sample_width == 4:
        samples = array("i")
        samples.frombytes(raw_frames)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported sample width: {sample_width}")

    float_samples: list[float] = []
    if channels <= 1:
        float_samples = [float(sample) for sample in samples]
    else:
        for index in range(0, len(samples), channels):
            chunk = samples[index : index + channels]
            float_samples.append(float(sum(chunk)) / len(chunk))

    if not float_samples:
        raise HTTPException(status_code=400, detail="Audio file contains no samples")

    return float_samples, sample_rate


def _frame_audio(samples: list[float], frame_size: int = 1024, hop_size: int = 512) -> pd.DataFrame:
    frames: list[dict[str, float]] = []
    if len(samples) < frame_size:
        samples = samples + [0.0] * (frame_size - len(samples))

    for start in range(0, len(samples) - frame_size + 1, hop_size):
        frame = samples[start : start + frame_size]
        abs_frame = [abs(value) for value in frame]
        zero_crossings = 0
        for left, right in zip(frame, frame[1:]):
            if (left >= 0 > right) or (left < 0 <= right):
                zero_crossings += 1

        rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        frames.append(
            {
                "mean_amplitude": statistics.fmean(frame),
                "std_amplitude": statistics.pstdev(frame) if len(frame) > 1 else 0.0,
                "variance_amplitude": statistics.pvariance(frame) if len(frame) > 1 else 0.0,
                "min_amplitude": min(frame),
                "max_amplitude": max(frame),
                "median_amplitude": statistics.median(frame),
                "mode_amplitude": statistics.fmean(frame),
                "range_amplitude": max(frame) - min(frame),
                "rms": rms,
                "zero_crossing_rate": zero_crossings / max(len(frame) - 1, 1),
                "mean_abs_amplitude": statistics.fmean(abs_frame),
            }
        )

    return pd.DataFrame(frames)


def _series_stats(series: pd.Series) -> dict[str, float]:
    values = [float(value) for value in series.dropna().tolist()]
    if not values:
        return {"mean": 0.0, "std": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "mode": 0.0, "range": 0.0}

    try:
        mode_value = statistics.mode(values)
    except statistics.StatisticsError:
        mode_value = values[0]

    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "variance": float(statistics.pvariance(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(statistics.median(values)),
        "mode": float(mode_value),
        "range": float(max(values) - min(values)),
    }


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

    mode_values: dict[str, float] = {}
    range_values: dict[str, float] = {}
    for column in df.columns:
        column_series = df[column]
        if pd.api.types.is_numeric_dtype(column_series):
            stats = _series_stats(column_series)
            mode_values[str(column)] = stats["mode"]
            range_values[str(column)] = stats["range"]

    response["mode"] = mode_values
    response["range"] = range_values
    response["allowed_values"] = {}
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
    samples, sample_rate = _load_wav_samples(audio_bytes)
    features = _frame_audio(samples)
    return _build_response(features, payload.audio_id, sample_rate, len(samples))


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "endpoint": "/analyze"}


@app.post("/preview")
def preview_audio(payload: AudioRequest) -> dict[str, Any]:
    audio_bytes = _decode_audio_bytes(payload.audio_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_path = Path(temp_file.name)

    try:
        samples, sample_rate = _load_wav_samples(temp_path.read_bytes())
        features = _frame_audio(samples)
        return _build_response(features, payload.audio_id, sample_rate, len(samples))
    finally:
        temp_path.unlink(missing_ok=True)
