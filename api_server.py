from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import librosa
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)


LOGGER = logging.getLogger("omnivoice.worker")
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
SOURCE_DIR = DATA_DIR / "voice_sources"
WORKER_PID_PATH = DATA_DIR / "worker.pid"
GRADIO_PID_PATH = DATA_DIR / "gradio.pid"
PROFILE_DIR = ROOT_DIR / "voice_library"
PROFILE_CATALOG_PATH = DATA_DIR / "profile_catalog.json"
LEGACY_SAMPLE_DIR = ROOT_DIR / "voice"
MODEL_ID = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
INTERNAL_TOKEN = os.environ.get("AUTO_YT_OMNIVOICE_TOKEN", "").strip()
IDLE_UNLOAD_SECONDS = max(
    60, int(os.environ.get("OMNIVOICE_IDLE_UNLOAD_SECONDS", "900"))
)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 250_000
LEGACY_MAX_CHUNK_CHARS = 700
ENGINE_REVISION = 3
DEFAULT_MAX_CHUNK_CHARS = 220
DEFAULT_TARGET_WORDS_PER_MINUTE = 145.0
DEFAULT_MIN_WORDS_PER_MINUTE = 105.0
DEFAULT_MAX_WORDS_PER_MINUTE = 240.0
PROFILE_ID_PATTERN = re.compile(r"^[a-f0-9-]{36}$")
SUPPORTED_SAMPLE_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}

for directory in (DATA_DIR, JOBS_DIR, SOURCE_DIR, PROFILE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _process_marker_is_alive(path: Path) -> bool:
    try:
        process_id = int(path.read_text(encoding="ascii").strip())
        os.kill(process_id, 0)
        return True
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return False


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_job_dir(job_id: str) -> Path:
    try:
        normalized = str(uuid.UUID(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job OmniVoice không hợp lệ.") from exc
    path = (JOBS_DIR / normalized).resolve()
    if path.parent != JOBS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Job OmniVoice không hợp lệ.")
    return path


def _profile_path(profile_id: str) -> Path:
    try:
        normalized = str(uuid.UUID(str(profile_id or "")))
    except ValueError as exc:
        raise ValueError("Profile ID OmniVoice không hợp lệ.") from exc
    if normalized != str(profile_id or "").casefold():
        raise ValueError("Profile ID OmniVoice không hợp lệ.")
    path = (PROFILE_DIR / f"{normalized}.pt").resolve()
    if path.parent != PROFILE_DIR.resolve():
        raise ValueError("Profile ID OmniVoice không hợp lệ.")
    return path


def _load_profile_catalog() -> dict[str, dict]:
    if not PROFILE_CATALOG_PATH.exists():
        return {}
    try:
        payload = json.loads(PROFILE_CATALOG_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("OmniVoice profile catalog is unreadable; rebuilding names lazily.")
        return {}


def _save_profile_catalog(catalog: dict[str, dict]) -> None:
    _atomic_json(PROFILE_CATALOG_PATH, catalog)


def _canonicalize_legacy_profiles(catalog: dict) -> list[dict]:
    """Expose pre-existing named .pt files through stable UUID profile IDs.

    The worker API never accepts file names or arbitrary paths.  A legacy
    profile is copied once to a UUID-named file derived from its resolved path;
    repeated syncs therefore return the same ID and cannot create duplicates.
    The original profile remains untouched for OmniVoice's existing UI.
    """
    profiles: list[dict] = []
    for original_path in sorted(PROFILE_DIR.glob("*.pt")):
        if PROFILE_ID_PATTERN.fullmatch(original_path.stem):
            continue
        profile_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"omnivoice-profile:{original_path.resolve()}",
            )
        )
        canonical_path = _profile_path(profile_id)
        if not canonical_path.exists():
            shutil.copy2(original_path, canonical_path)
            
        metadata = catalog.get(profile_id) or {}
        profiles.append(
            {
                "id": profile_id,
                "name": original_path.stem,
                "status": "active",
                "kind": "profile",
                "legacy_source": True,
                "settings": metadata.get("settings", {}),
            }
        )
    return profiles


def _read_manifest(job_dir: Path) -> dict:
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy job OmniVoice.")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Manifest OmniVoice bị hỏng.") from exc


def _split_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    # OmniVoice Guidelines: Strict Sentence Chunking by punctuation [.!?\n]
    normalized = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if not normalized:
        raise ValueError("Nội dung tạo audio không được để trống.")
    if len(normalized) > MAX_TEXT_CHARS:
        raise ValueError(f"Nội dung vượt quá {MAX_TEXT_CHARS:,} ký tự.")
    chunks: list[str] = []
    # Tách đoạn văn thành từng câu dựa trên dấu chấm, chấm hỏi, chấm than, hoặc xuống dòng
    raw_sentences = re.split(r'(?<=[.!?\n])\s+', normalized)
    for s in raw_sentences:
        s = s.strip()
        if s:
            chunks.append(s)
            
    return chunks


def _bounded_number(
    value: object,
    default: float,
    minimum: float,
    maximum: float,
    label: str,
) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} không hợp lệ.") from exc
    if not np.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"{label} phải nằm trong khoảng {minimum:g}–{maximum:g}.")
    return normalized


def _normalize_new_job_settings(raw_settings: object) -> dict:
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    minimum_wpm = _bounded_number(
        settings.get("min_words_per_minute", DEFAULT_MIN_WORDS_PER_MINUTE),
        DEFAULT_MIN_WORDS_PER_MINUTE,
        60,
        220,
        "Tốc độ tối thiểu",
    )
    maximum_wpm = _bounded_number(
        settings.get("max_words_per_minute", DEFAULT_MAX_WORDS_PER_MINUTE),
        DEFAULT_MAX_WORDS_PER_MINUTE,
        120,
        360,
        "Tốc độ tối đa",
    )
    target_wpm = _bounded_number(
        settings.get("target_words_per_minute", DEFAULT_TARGET_WORDS_PER_MINUTE),
        DEFAULT_TARGET_WORDS_PER_MINUTE,
        minimum_wpm,
        maximum_wpm,
        "Tốc độ mục tiêu",
    )
    return {
        "engine_revision": max(ENGINE_REVISION, int(settings.get("engine_revision") or ENGINE_REVISION)),
        "target_words_per_minute": target_wpm,
        "min_words_per_minute": minimum_wpm,
        "max_words_per_minute": maximum_wpm,
        "max_chunk_chars": int(_bounded_number(
            settings.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS),
            DEFAULT_MAX_CHUNK_CHARS,
            120,
            400,
            "Độ dài chunk",
        )),
        "pause_ms": int(_bounded_number(
            settings.get("pause_ms", 150), 150, 0, 1000, "Khoảng nghỉ"
        )),
        # Bắt buộc dùng 32 để đạt chất lượng phòng thu, bỏ qua mọi yêu cầu cũ (ví dụ 16)
        "num_step": 32,
        # Denoising a cloned voice can discard the beginning of the requested
        # speech. Keep it disabled until the upstream model provides a safe
        # voice-cloning implementation for this option.
        "denoise": False,
        "audio_chunk_duration": _bounded_number(
            settings.get("audio_chunk_duration", 12), 12, 5, 30, "Thời lượng chunk nội bộ"
        ),
        "audio_chunk_threshold": _bounded_number(
            settings.get("audio_chunk_threshold", 24), 24, 10, 60, "Ngưỡng chunk nội bộ"
        ),
    }


def _is_quality_managed(settings: object) -> bool:
    return isinstance(settings, dict) and int(settings.get("engine_revision") or 1) >= ENGINE_REVISION


def _build_inference_voice_prompt(profile: VoiceClonePrompt) -> VoiceClonePrompt:
    """Giữ nguyên ref_text để OmniVoice có text mồi, chống ảo giác."""
    return VoiceClonePrompt(
        ref_audio_tokens=profile.ref_audio_tokens,
        ref_text=profile.ref_text,
        ref_rms=profile.ref_rms,
    )


def _validate_generated_audio(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    settings: dict,
    *,
    extra_allowance_seconds: float = 4.0,
) -> float:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0 or sample_rate <= 0:
        raise RuntimeError("OmniVoice trả về audio rỗng.")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("OmniVoice trả về audio chứa dữ liệu không hợp lệ.")
    duration_seconds = samples.size / sample_rate
    if float(np.sqrt(np.mean(np.square(samples)))) < 1e-5:
        raise RuntimeError("OmniVoice trả về audio gần như im lặng.")

    word_count = len(str(text or "").split())
    if word_count:
        minimum_wpm = float(settings.get("min_words_per_minute", DEFAULT_MIN_WORDS_PER_MINUTE))
        maximum_wpm = float(settings.get("max_words_per_minute", DEFAULT_MAX_WORDS_PER_MINUTE))
        minimum_duration = max(1.0, word_count * 60.0 / maximum_wpm - 2.0)
        maximum_duration = max(
            8.0,
            word_count * 60.0 / minimum_wpm + extra_allowance_seconds,
        )
        if duration_seconds < minimum_duration:
            raise RuntimeError(
                "OmniVoice trả về audio quá ngắn so với nội dung; job đã dừng để tránh mất lời."
            )
        if duration_seconds > maximum_duration:
            raise RuntimeError(
                "OmniVoice trả về audio quá dài so với nội dung; job đã dừng để tránh lặp hoặc sai lời."
            )
    return duration_seconds


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        sample_rate = source.getframerate()
    if frames <= 0 or sample_rate <= 0:
        raise ValueError("File WAV không có dữ liệu âm thanh.")
    return frames / sample_rate


class ModelRuntime:
    def __init__(self) -> None:
        self._model: OmniVoice | None = None
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._busy = False
        self._state = "sleeping"
        self._error = ""

    def acquire(self) -> OmniVoice:
        with self._lock:
            if _process_marker_is_alive(GRADIO_PID_PATH):
                raise RuntimeError(
                    "OmniVoice Gradio đang giữ model GPU. Hãy dừng cổng thử nghiệm "
                    "8001 trước khi chạy job Auto_YT."
                )
            self._busy = True
            self._error = ""
            if self._model is None:
                self._state = "loading"
                try:
                    self._model = OmniVoice.from_pretrained(
                        MODEL_ID,
                        device_map="cuda",
                        dtype=torch.float16,
                    )
                except Exception as exc:
                    self._busy = False
                    self._state = "error"
                    self._error = str(exc)
                    raise
            self._state = "busy"
            self._last_used = time.monotonic()
            return self._model

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._last_used = time.monotonic()
            self._state = "ready"

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._model is None or self._busy:
                return False
            if time.monotonic() - self._last_used < IDLE_UNLOAD_SECONDS:
                return False
            self._model = None
            self._state = "sleeping"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        LOGGER.info("OmniVoice model unloaded after idle timeout.")
        return True

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "model_loaded": self._model is not None,
                "busy": self._busy,
                "error": self._error,
                "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
            }


RUNTIME = ModelRuntime()
JOB_QUEUE: queue.Queue[str] = queue.Queue()
QUEUED_IDS: set[str] = set()
QUEUE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


def _queue_job(job_id: str) -> None:
    with QUEUE_LOCK:
        if job_id in QUEUED_IDS:
            return
        QUEUED_IDS.add(job_id)
        JOB_QUEUE.put(job_id)


def _merge_chunks(job_dir: Path, manifest: dict) -> Path:
    chunk_paths = [job_dir / chunk["file"] for chunk in manifest["chunks"]]
    if not chunk_paths or any(not path.exists() for path in chunk_paths):
        raise RuntimeError("Chưa tạo đủ chunk audio để ghép.")
    pause_ms = max(0, min(int(manifest.get("settings", {}).get("pause_ms", 250)), 2000))
    audio_parts: list[np.ndarray] = []
    sample_rate: int | None = None
    for index, path in enumerate(chunk_paths):
        audio, current_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.size == 0:
            raise RuntimeError(f"Chunk audio {index + 1} bị rỗng.")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = int(current_rate)
        if int(current_rate) != sample_rate:
            raise RuntimeError("Các chunk audio có sample rate không đồng nhất.")
            
        # Thêm fade in/out 10ms để khử tiếng 'bụp' khi ghép (pop/click noise)
        fade_samples = int(sample_rate * 0.01) # 10ms
        audio_length = len(audio)
        if audio_length > fade_samples * 2:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
            
        audio_parts.append(audio)
        if pause_ms and index < len(chunk_paths) - 1:
            audio_parts.append(np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32))
    final_path = job_dir / "final.wav"
    merged_audio = np.concatenate(audio_parts)
    if _is_quality_managed(manifest.get("settings")):
        pause_seconds = pause_ms * max(0, len(chunk_paths) - 1) / 1000.0
        _validate_generated_audio(
            merged_audio,
            int(sample_rate),
            " ".join(chunk["text"] for chunk in manifest["chunks"]),
            manifest["settings"],
            extra_allowance_seconds=4.0 + pause_seconds,
        )
    sf.write(str(final_path), merged_audio, sample_rate, subtype="PCM_16")
    _wav_duration(final_path)
    return final_path


def _process_job(job_id: str) -> None:
    job_dir = _safe_job_dir(job_id)
    manifest = _read_manifest(job_dir)
    if manifest.get("status") == "completed" and (job_dir / "final.wav").exists():
        return
    profile_path = _profile_path(manifest.get("voice_id", ""))
    if not profile_path.exists():
        raise RuntimeError("Profile giọng OmniVoice không còn tồn tại.")
    manifest["status"] = "processing"
    manifest["error"] = ""
    manifest["updated_at"] = time.time()
    _atomic_json(job_dir / "manifest.json", manifest)
    model = RUNTIME.acquire()
    try:
        stored_voice_prompt = VoiceClonePrompt.load(str(profile_path))
        voice_prompt = _build_inference_voice_prompt(stored_voice_prompt)
        settings = manifest.get("settings") or {}
        quality_managed = _is_quality_managed(settings)
        generation_config = OmniVoiceGenerationConfig(
            num_step=32, # Cưỡng chế tuyệt đối 32 bước sinh
            denoise=False,
            preprocess_prompt=True,
            postprocess_output=True,
            audio_chunk_duration=float(settings.get("audio_chunk_duration", 15.0)),
            audio_chunk_threshold=float(settings.get("audio_chunk_threshold", 30.0)),
        )
        for index, chunk in enumerate(manifest["chunks"]):
            # The cancel endpoint updates the manifest from another thread.
            latest = _read_manifest(job_dir)
            manifest["cancel_requested"] = bool(latest.get("cancel_requested"))
            if manifest.get("cancel_requested"):
                manifest["status"] = "canceled"
                break
            chunk_path = job_dir / chunk["file"]
            if quality_managed:
                expected_text_hash = hashlib.sha256(
                    chunk["text"].encode("utf-8")
                ).hexdigest()
                if chunk.get("text_hash") != expected_text_hash:
                    raise RuntimeError(
                        f"Nội dung chunk {index + 1} không khớp manifest; job đã dừng an toàn."
                    )
            if chunk.get("status") == "completed" and chunk_path.exists():
                try:
                    if quality_managed:
                        existing_audio, existing_rate = sf.read(
                            str(chunk_path), dtype="float32", always_2d=False
                        )
                        _validate_generated_audio(
                            existing_audio,
                            int(existing_rate),
                            chunk["text"],
                            settings,
                        )
                    else:
                        _wav_duration(chunk_path)
                    continue
                except (OSError, RuntimeError, ValueError, wave.Error):
                    chunk_path.unlink(missing_ok=True)
            target_duration = None
            if quality_managed:
                pass  # Disabled forcing duration to prevent voice distortion
            generated = model.generate(
                text=chunk["text"],
                language="Vietnamese",
                voice_clone_prompt=voice_prompt,
                duration=target_duration,
                generation_config=generation_config,
            )
            raw_audio = np.asarray(generated[0], dtype=np.float32)
            if raw_audio.size == 0:
                raise RuntimeError(f"OmniVoice trả về chunk rỗng ở phần {index + 1}.")
                
            # Cắt bỏ khoảng lặng (silence) dư thừa ở đầu và cuối chunk
            # Giúp loại bỏ tiếng bụp sinh ra ở cuối khoảng lặng và giảm thời gian chờ
            audio, _ = librosa.effects.trim(raw_audio, top_db=40)
            if audio.size == 0:
                audio = raw_audio # Fallback if completely trimmed

            chunk_duration = (
                _validate_generated_audio(
                    audio,
                    int(model.sampling_rate),
                    chunk["text"],
                    settings,
                )
                if quality_managed
                else audio.size / int(model.sampling_rate)
            )
            sf.write(str(chunk_path), audio, int(model.sampling_rate), subtype="PCM_16")
            _wav_duration(chunk_path)
            chunk["status"] = "completed"
            chunk["duration_seconds"] = chunk_duration
            manifest["completed_chunks"] = index + 1
            manifest["updated_at"] = time.time()
            _atomic_json(job_dir / "manifest.json", manifest)
        if manifest.get("status") != "canceled":
            final_path = _merge_chunks(job_dir, manifest)
            manifest["status"] = "completed"
            manifest["audio_file"] = final_path.name
            manifest["duration_seconds"] = _wav_duration(final_path)
        manifest["updated_at"] = time.time()
        _atomic_json(job_dir / "manifest.json", manifest)
    finally:
        RUNTIME.release()


def _worker_loop() -> None:
    while not STOP_EVENT.is_set():
        try:
            job_id = JOB_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            _process_job(job_id)
        except Exception as exc:
            LOGGER.exception("OmniVoice job %s failed", job_id)
            try:
                job_dir = _safe_job_dir(job_id)
                manifest = _read_manifest(job_dir)
                manifest["status"] = "failed"
                manifest["error"] = str(exc)
                manifest["updated_at"] = time.time()
                _atomic_json(job_dir / "manifest.json", manifest)
            except Exception:
                LOGGER.exception("Could not persist failure for OmniVoice job %s", job_id)
        finally:
            with QUEUE_LOCK:
                QUEUED_IDS.discard(job_id)
            JOB_QUEUE.task_done()


def _idle_loop() -> None:
    while not STOP_EVENT.wait(30):
        RUNTIME.unload_if_idle()


def _recover_jobs() -> None:
    for manifest_path in JOBS_DIR.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") in {"queued", "processing"}:
                manifest["status"] = "queued"
                manifest["error"] = ""
                _atomic_json(manifest_path, manifest)
                _queue_job(manifest["id"])
        except Exception:
            LOGGER.exception("Could not recover %s", manifest_path)


def _require_token(authorization: str = Header(default="")) -> None:
    if not INTERNAL_TOKEN:
        raise HTTPException(status_code=503, detail="OmniVoice internal token is missing.")
    scheme, _, supplied = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not secrets.compare_digest(
        supplied.strip(), INTERNAL_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Invalid internal token.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    STOP_EVENT.clear()
    WORKER_PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    _recover_jobs()
    worker = threading.Thread(target=_worker_loop, name="omnivoice-gpu-worker", daemon=True)
    idle_worker = threading.Thread(target=_idle_loop, name="omnivoice-idle-worker", daemon=True)
    worker.start()
    idle_worker.start()
    try:
        yield
    finally:
        STOP_EVENT.set()
        WORKER_PID_PATH.unlink(missing_ok=True)


app = FastAPI(title="OmniVoice Worker", version="1.0", lifespan=lifespan)


class TTSJobRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    voice_id: str
    request_hash: str = Field(min_length=64, max_length=64)
    settings: dict[str, Any] = Field(default_factory=dict)

class VoiceSettingsUpdateRequest(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


@app.get("/health", dependencies=[Depends(_require_token)])
def health() -> dict:
    runtime = RUNTIME.status()
    return {
        "ok": runtime["state"] != "error",
        **runtime,
        "queued_jobs": JOB_QUEUE.qsize(),
    }


@app.patch("/v1/voices/{voice_id}/settings", dependencies=[Depends(_require_token)])
def update_voice_settings(voice_id: str, request: VoiceSettingsUpdateRequest) -> dict:
    profile_path = _profile_path(voice_id)
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile không tồn tại.")
        
    catalog = _load_profile_catalog()
    if voice_id not in catalog:
        catalog[voice_id] = {}
        
    current_settings = catalog[voice_id].get("settings", {})
    # Update with new settings
    for k, v in request.settings.items():
        if v is None:
            current_settings.pop(k, None)
        else:
            current_settings[k] = v
            
    catalog[voice_id]["settings"] = current_settings
    _save_profile_catalog(catalog)
    
    return {"id": voice_id, "settings": current_settings}

@app.get("/v1/voices", dependencies=[Depends(_require_token)])
def list_voices() -> dict:
    catalog = _load_profile_catalog()
    legacy_profiles = _canonicalize_legacy_profiles(catalog)
    legacy_ids = {profile["id"] for profile in legacy_profiles}
    profiles = list(legacy_profiles)
    for profile_path in sorted(PROFILE_DIR.glob("*.pt")):
        profile_id = profile_path.stem
        if PROFILE_ID_PATTERN.fullmatch(profile_id) and profile_id not in legacy_ids:
            metadata = catalog.get(profile_id) or {}
            profiles.append(
                {
                    "id": profile_id,
                    "name": str(metadata.get("name") or profile_id),
                    "status": "active",
                    "kind": "profile",
                    "validation_duration_seconds": metadata.get(
                        "validation_duration_seconds"
                    ),
                    "settings": metadata.get("settings", {}),
                }
            )
    samples = []
    for sample_dir in (LEGACY_SAMPLE_DIR, SOURCE_DIR):
        if not sample_dir.exists():
            continue
        for sample_path in sorted(sample_dir.iterdir()):
            if not sample_path.is_file() or sample_path.suffix.casefold() not in SUPPORTED_SAMPLE_SUFFIXES:
                continue
            try:
                info = sf.info(str(sample_path))
                samples.append(
                    {
                        "id": hashlib.sha256(str(sample_path.resolve()).encode()).hexdigest()[:16],
                        "name": sample_path.name,
                        "status": "sample_pending",
                        "kind": "sample",
                        "duration_seconds": float(info.duration),
                        "sample_rate": int(info.samplerate),
                        "channels": int(info.channels),
                    }
                )
            except (OSError, RuntimeError):
                LOGGER.warning("Could not inspect sample %s", sample_path)
    return {"profiles": profiles, "samples": samples}


@app.post("/v1/voices/clone", dependencies=[Depends(_require_token)])
async def clone_voice(
    name: str = Form(...),
    start_seconds: float = Form(...),
    end_seconds: float = Form(...),
    reference_text: str = Form(..., min_length=2, max_length=2000),
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "").suffix.casefold()
    if suffix not in SUPPORTED_SAMPLE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ WAV, MP3, FLAC hoặc OGG.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File mẫu rỗng hoặc vượt quá 100 MB.")
    upload_path = SOURCE_DIR / f"upload-{uuid.uuid4().hex}{suffix}"
    upload_path.write_bytes(contents)
    try:
        audio, sample_rate = sf.read(str(upload_path), dtype="float32", always_2d=True)
        if audio.size == 0 or sample_rate <= 0:
            raise ValueError("File mẫu không có dữ liệu âm thanh.")
        duration = len(audio) / sample_rate
        if start_seconds < 0 or end_seconds <= start_seconds or end_seconds > duration + 0.01:
            raise ValueError("Khoảng cắt mẫu không hợp lệ.")
        selected_duration = end_seconds - start_seconds
        if selected_duration < 3 or selected_duration > 10:
            raise ValueError("Đoạn mẫu phải dài từ 3 đến 10 giây.")
        start_frame = int(start_seconds * sample_rate)
        end_frame = min(len(audio), int(end_seconds * sample_rate))
        mono = audio[start_frame:end_frame].mean(axis=1)
        source_id = str(uuid.uuid4())
        cropped_path = SOURCE_DIR / f"{source_id}.wav"
        sf.write(str(cropped_path), mono, sample_rate, subtype="PCM_16")
        _wav_duration(cropped_path)
        model = RUNTIME.acquire()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(cropped_path),
                ref_text=reference_text.strip(),
            )
            profile_id = str(uuid.uuid4())
            # A profile is exposed as active only after the same prompt has
            # successfully synthesized a real sample.  This catches invalid
            # references/model errors before Auto_YT can enqueue long videos.
            validation = model.generate(
                text="Xin chào, đây là bản kiểm tra giọng đọc.",
                language="Vietnamese",
                voice_clone_prompt=_build_inference_voice_prompt(prompt),
                generation_config=OmniVoiceGenerationConfig(
                    num_step=32,
                    denoise=False,
                    preprocess_prompt=True,
                    postprocess_output=True,
                ),
            )
            validation_audio = np.asarray(validation[0], dtype=np.float32)
            if validation_audio.size == 0:
                raise RuntimeError("OmniVoice trả về bản thử giọng rỗng.")
            validation_path = SOURCE_DIR / f"{profile_id}-preview.wav"
            sf.write(
                str(validation_path),
                validation_audio,
                int(model.sampling_rate),
                subtype="PCM_16",
            )
            validation_duration = _wav_duration(validation_path)
            prompt.save(str(_profile_path(profile_id)))
            catalog = _load_profile_catalog()
            catalog[profile_id] = {
                "name": name.strip() or profile_id,
                "source_id": source_id,
                "validation_duration_seconds": validation_duration,
                "created_at": time.time(),
            }
            _save_profile_catalog(catalog)
        finally:
            RUNTIME.release()
        return {
            "id": profile_id,
            "name": name.strip() or profile_id,
            "status": "active",
            "source_id": source_id,
            "duration_seconds": selected_duration,
            "validation_duration_seconds": validation_duration,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        upload_path.unlink(missing_ok=True)


@app.post("/v1/jobs", dependencies=[Depends(_require_token)])
def create_job(request: TTSJobRequest) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", request.request_hash.casefold()):
        raise HTTPException(status_code=400, detail="Request hash không hợp lệ.")
    try:
        profile_path = _profile_path(request.voice_id)
        
        # Merge order: Client Request > Voice Profile Defaults > System Defaults
        catalog = _load_profile_catalog()
        voice_metadata = catalog.get(request.voice_id) or {}
        voice_settings = voice_metadata.get("settings", {})
        merged_settings = {**voice_settings, **(request.settings or {})}
        
        settings = _normalize_new_job_settings(merged_settings)
        chunks = _split_text(
            request.text,
            max_chars=int(settings["max_chunk_chars"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not profile_path.exists():
        raise HTTPException(status_code=400, detail="Profile giọng OmniVoice không tồn tại.")
    for manifest_path in JOBS_DIR.glob("*/manifest.json"):
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("request_hash") == request.request_hash:
                if existing.get("status") in {"queued", "processing"}:
                    _queue_job(existing["id"])
                return existing
        except (OSError, json.JSONDecodeError):
            continue
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    now = time.time()
    manifest = {
        "id": job_id,
        "status": "queued",
        "request_hash": request.request_hash,
        "voice_id": request.voice_id,
        "settings": settings,
        "text_hash": hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
        "chunks": [
            {
                "index": index,
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "file": f"chunk-{index:05d}.wav",
                "status": "pending",
            }
            for index, text in enumerate(chunks)
        ],
        "completed_chunks": 0,
        "cancel_requested": False,
        "audio_file": "",
        "error": "",
        "created_at": now,
        "updated_at": now,
    }
    _atomic_json(job_dir / "manifest.json", manifest)
    _queue_job(job_id)
    return manifest


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(_require_token)])
def get_job(job_id: str) -> dict:
    return _read_manifest(_safe_job_dir(job_id))


@app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(_require_token)])
def cancel_job(job_id: str) -> dict:
    job_dir = _safe_job_dir(job_id)
    manifest = _read_manifest(job_dir)
    if manifest.get("status") in {"completed", "failed", "canceled"}:
        return manifest
    manifest["cancel_requested"] = True
    if manifest.get("status") == "queued":
        manifest["status"] = "canceled"
    manifest["updated_at"] = time.time()
    _atomic_json(job_dir / "manifest.json", manifest)
    return manifest


@app.post("/v1/jobs/{job_id}/recover", dependencies=[Depends(_require_token)])
def recover_job(job_id: str) -> dict:
    job_dir = _safe_job_dir(job_id)
    manifest = _read_manifest(job_dir)
    if manifest.get("status") == "completed" and (job_dir / "final.wav").exists():
        return manifest
    manifest["status"] = "queued"
    manifest["cancel_requested"] = False
    manifest["error"] = ""
    manifest["updated_at"] = time.time()
    _atomic_json(job_dir / "manifest.json", manifest)
    _queue_job(job_id)
    return manifest


@app.get("/v1/jobs/{job_id}/audio", dependencies=[Depends(_require_token)])
def download_audio(job_id: str):
    job_dir = _safe_job_dir(job_id)
    manifest = _read_manifest(job_dir)
    audio_path = job_dir / "final.wav"
    if manifest.get("status") != "completed" or not audio_path.exists():
        raise HTTPException(status_code=409, detail="Audio OmniVoice chưa hoàn tất.")
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{job_id}.wav")
