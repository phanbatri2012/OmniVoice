import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import api_server


class OmniVoiceWorkerTests(unittest.TestCase):
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

    def test_legacy_manifest_does_not_enable_new_quality_policy(self):
        self.assertFalse(api_server._is_quality_managed({"pause_ms": 250}))


if __name__ == "__main__":
    unittest.main()
