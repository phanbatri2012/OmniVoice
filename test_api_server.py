import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import api_server


class OmniVoiceWorkerTests(unittest.TestCase):
    def test_runtime_status_does_not_block_while_model_is_loading(self):
        runtime = api_server.ModelRuntime()
        load_started = threading.Event()
        allow_load_to_finish = threading.Event()
        model = object()

        def slow_load(*_args, **_kwargs):
            load_started.set()
            self.assertTrue(allow_load_to_finish.wait(timeout=5))
            return model

        with (
            patch.object(api_server, "_process_marker_is_alive", return_value=False),
            patch.object(api_server.OmniVoice, "from_pretrained", side_effect=slow_load),
        ):
            loader = threading.Thread(target=runtime.acquire)
            loader.start()
            self.assertTrue(load_started.wait(timeout=2))

            started_at = time.monotonic()
            status = runtime.status()
            elapsed = time.monotonic() - started_at

            allow_load_to_finish.set()
            loader.join(timeout=5)

        self.assertFalse(loader.is_alive())
        self.assertLess(elapsed, 0.5)
        self.assertEqual(status["state"], "loading")
        self.assertTrue(status["busy"])
        self.assertFalse(status["model_loaded"])

    def test_natural_chunking_is_bounded_and_lossless(self):
        source = ("Một câu kể chuyện có dấu câu. " * 80).strip()
        chunks = api_server._split_text(source, max_chars=120)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(" ".join(chunks).replace("  ", " "), source)

    def test_legacy_profile_sync_is_idempotent_and_uses_safe_uuid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_dir = Path(temporary_directory)
            legacy = profile_dir / "dinh_doan.pt"
            legacy.write_bytes(b"legacy-profile")
            with patch.object(api_server, "PROFILE_DIR", profile_dir):
                first = api_server._canonicalize_legacy_profiles({})
                second = api_server._canonicalize_legacy_profiles({})

            self.assertEqual(first, second)
            self.assertEqual(len(first), 1)
            profile_id = first[0]["id"]
            self.assertRegex(profile_id, api_server.PROFILE_ID_PATTERN)
            self.assertEqual((profile_dir / f"{profile_id}.pt").read_bytes(), b"legacy-profile")
            self.assertEqual(legacy.read_bytes(), b"legacy-profile")

    def test_profile_api_rejects_paths(self):
        for unsafe_id in ("../voice", "voice/name", "voice\\name"):
            with self.assertRaises(ValueError):
                api_server._profile_path(unsafe_id)

    def test_new_jobs_receive_bounded_quality_settings(self):
        settings = api_server._normalize_new_job_settings({})

        self.assertEqual(settings["engine_revision"], 3)
        self.assertEqual(settings["max_chunk_chars"], 220)
        self.assertEqual(settings["target_words_per_minute"], 145)
        self.assertFalse(settings["denoise"])

    def test_requested_denoise_is_disabled_for_voice_cloning(self):
        settings = api_server._normalize_new_job_settings({"denoise": True})

        self.assertFalse(settings["denoise"])

    def test_poetry_prosody_is_opt_in_and_bounded(self):
        standard = api_server._normalize_new_job_settings({})
        poetry = api_server._normalize_new_job_settings(
            {
                "prosody_mode": api_server.VIETNAMESE_POETRY_PROSODY,
                "poetry_line_pause_ms": 550,
                "poetry_couplet_pause_ms": 1100,
            }
        )

        self.assertNotIn("prosody_mode", standard)
        self.assertEqual(poetry["prosody_mode"], api_server.VIETNAMESE_POETRY_PROSODY)
        self.assertEqual(poetry["poetry_line_pause_ms"], 550)
        self.assertEqual(poetry["poetry_couplet_pause_ms"], 1100)
        with self.assertRaisesRegex(ValueError, "ngữ điệu"):
            api_server._normalize_new_job_settings({"prosody_mode": "global_poetry"})

    def test_seven_syllable_poetry_preserves_author_punctuation(self):
        settings = api_server._normalize_new_job_settings(
            {"prosody_mode": api_server.VIETNAMESE_POETRY_PROSODY}
        )
        chunks = api_server._split_vietnamese_poetry(
            "TỨC CẢNH PÁC BÓ.\n\n"
            "Sáng ra bờ suối, tối vào hang\n"
            "Cháo bẹ, rau măng vẫn sẵn sàng\n"
            "Bàn đá chông chênh dịch sử Đảng\n"
            "Cuộc đời cách mạng thật là sang",
            settings,
        )

        self.assertEqual(chunks[0]["role"], "title")
        self.assertTrue(all(chunk["poem_form"] == "seven_syllable" for chunk in chunks))
        self.assertEqual(chunks[1]["rhythm"], "4/3")
        self.assertEqual(chunks[2]["rhythm"], "2/2/3")
        self.assertIn("Cháo bẹ, rau măng, vẫn sẵn sàng", chunks[2]["text"])
        self.assertEqual(chunks[1]["pause_after_ms"], 550)
        self.assertEqual(chunks[2]["pause_after_ms"], 1100)
        self.assertEqual(chunks[-1]["pause_after_ms"], 0)

    def test_luc_bat_uses_line_and_couplet_pauses(self):
        settings = api_server._normalize_new_job_settings(
            {"prosody_mode": api_server.VIETNAMESE_POETRY_PROSODY}
        )
        chunks = api_server._split_vietnamese_poetry(
            "Bắc Nam như cội với cành\n"
            "anh em ruột thịt đấu tranh một lòng\n"
            "rồi đây thống nhất thành công\n"
            "Bắc Nam ta lại vui chung một nhà",
            settings,
        )

        self.assertEqual(chunks[0]["poem_form"], "luc_bat")
        self.assertEqual(chunks[0]["rhythm"], "2/2/2")
        self.assertEqual(chunks[1]["rhythm"], "2/2/2/2")
        self.assertEqual(chunks[0]["pause_after_ms"], 550)
        self.assertEqual(chunks[1]["pause_after_ms"], 1100)

    def test_merge_uses_per_chunk_poetry_pauses(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory)
            sample_rate = 8_000
            tone = np.ones(sample_rate, dtype=np.float32) * 0.05
            sf.write(job_dir / "chunk-00000.wav", tone, sample_rate)
            sf.write(job_dir / "chunk-00001.wav", tone, sample_rate)
            manifest = {
                "settings": {"pause_ms": 150},
                "chunks": [
                    {"file": "chunk-00000.wav", "text": "dòng một", "pause_after_ms": 700},
                    {"file": "chunk-00001.wav", "text": "dòng hai", "pause_after_ms": 0},
                ],
            }

            final_path = api_server._merge_chunks(job_dir, manifest)
            info = sf.info(final_path)

        self.assertAlmostEqual(info.duration, 2.7, places=2)

    def test_inference_prompt_preserves_reference_transcript(self):
        tokens = torch.ones((8, 20), dtype=torch.long)
        stored = api_server.VoiceClonePrompt(
            ref_audio_tokens=tokens,
            ref_text="Nội dung mẫu không được để trống.",
            ref_rms=0.05,
        )

        inference = api_server._build_inference_voice_prompt(stored)

        self.assertIs(inference.ref_audio_tokens, tokens)
        self.assertEqual(inference.ref_text, "Nội dung mẫu không được để trống.")
        self.assertEqual(inference.ref_rms, stored.ref_rms)
        self.assertEqual(stored.ref_text, "Nội dung mẫu không được để trống.")

    def test_quality_gate_rejects_systematically_slow_audio(self):
        settings = api_server._normalize_new_job_settings({})
        audio = np.ones(24_000 * 90, dtype=np.float32) * 0.05

        with self.assertRaisesRegex(RuntimeError, "quá dài"):
            api_server._validate_generated_audio(
                audio,
                24_000,
                "một đoạn nội dung có khoảng mười từ để đọc thử",
                settings,
            )

    def test_quality_gate_accepts_short_dialogue_phrase(self):
        settings = api_server._normalize_new_job_settings({})
        # 4 words generated in 0.55s at 24kHz
        audio = np.ones(int(24_000 * 0.55), dtype=np.float32) * 0.05
        duration = api_server._validate_generated_audio(
            audio,
            24_000,
            "Toàn bộ mảnh đất?",
            settings,
        )
        self.assertAlmostEqual(duration, 0.55, places=2)

    def test_legacy_manifest_does_not_enable_new_quality_policy(self):
        self.assertFalse(api_server._is_quality_managed({"pause_ms": 250}))


if __name__ == "__main__":
    unittest.main()
