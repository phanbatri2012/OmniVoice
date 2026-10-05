import sys
import os
import re
import time
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import soundfile as sf
import torch
import librosa

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

def main():
    ref_audio_path = r"E:\yt\Tool\omnivoice\voice\nam_trung_nien_3\nam_trung_nien_3.WAV"
    output_path = r"E:\yt\Tool\omnivoice\output\nam_trung_nien_3_cloned.wav"
    profile_pt_path = r"E:\yt\Tool\omnivoice\voice_library\nam_trung_nien_3.pt"
    temp_ref_crop_path = r"E:\yt\Tool\omnivoice\voice\nam_trung_nien_3\nam_trung_nien_3_ref_7s.wav"

    target_text = (
        "Chào bác quan sâm và quý khán giả. "
        "Hôm đó là một buổi chiều mưa tầm tã. "
        "Tôi kết thúc chuyên án sớm hơn 3 ngày nên muốn về nhà tạo sự bất ngờ cho vợ và mẹ."
    )

    # 1. Prepare 6.8s reference audio & exact transcript (within 3-10s standard)
    ref_text = "cháu nằm trên giường nghe vợ và mẹ vợ bàn chuyện bán căn nhà rút tiền rồi thúc nhau lấy chữ ký của cháu."
    
    # Read and crop reference audio to 0.0s - 6.8s
    raw_ref, ref_sr = sf.read(ref_audio_path, dtype="float32", always_2d=True)
    crop_samples = int(6.8 * ref_sr)
    mono_ref = raw_ref[:crop_samples].mean(axis=1)
    sf.write(temp_ref_crop_path, mono_ref, ref_sr, subtype="PCM_16")

    print("=== [OmniVoice Voice Clone - nam_trung_nien_3] ===", flush=True)
    print(f"Ref Audio (Cropped 6.8s): {temp_ref_crop_path}", flush=True)
    print(f"Ref Text: {ref_text}\n", flush=True)

    # 2. Strict sentence chunking by punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', target_text.strip()) if s.strip()]
    print(f"Target Sentences ({len(sentences)}):", flush=True)
    for idx, s in enumerate(sentences, 1):
        print(f"  [{idx}] {s}", flush=True)

    # 3. Model & Generation Config
    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        denoise=False,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    print("\n[1/5] Loading OmniVoice model on CUDA float16...", flush=True)
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
    )
    sample_rate = int(model.sampling_rate)
    print(f"Loaded model in {time.time() - t0:.2f}s | Sample rate: {sample_rate} Hz", flush=True)

    # 4. Create & Save Voice Clone Prompt Profile for Tool Scanning
    print("\n[2/5] Extracting & registering voice prompt profile...", flush=True)
    t1 = time.time()
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=temp_ref_crop_path,
        ref_text=ref_text,
        preprocess_prompt=True,
    )
    Path(profile_pt_path).parent.mkdir(parents=True, exist_ok=True)
    voice_prompt.save(profile_pt_path)
    print(f"Voice prompt saved to {profile_pt_path} in {time.time() - t1:.2f}s", flush=True)

    # 5. Generate audio chunk by chunk
    print("\n[3/5] Synthesizing sentences...", flush=True)
    audio_parts = []
    pause_samples = int(sample_rate * 0.25)  # 250ms pause
    fade_samples = int(sample_rate * 0.01)   # 10ms crossfade

    for idx, sentence in enumerate(sentences, 1):
        st = time.time()
        generated = model.generate(
            text=sentence,
            language="Vietnamese",
            voice_clone_prompt=voice_prompt,
            duration=None,
            generation_config=gen_config,
        )
        raw_audio = np.asarray(generated[0], dtype=np.float32)
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)

        # Trim boundary silence
        trimmed_audio, _ = librosa.effects.trim(raw_audio, top_db=40)
        if trimmed_audio.size == 0:
            trimmed_audio = raw_audio

        # Apply 10ms fade-in/fade-out to avoid click/pop artifacts
        if len(trimmed_audio) > fade_samples * 2:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            trimmed_audio[:fade_samples] *= fade_in
            trimmed_audio[-fade_samples:] *= fade_out

        audio_parts.append(trimmed_audio)

        # Add pause between sentences
        if idx < len(sentences):
            silence = np.zeros(pause_samples, dtype=np.float32)
            audio_parts.append(silence)

        chunk_dur = len(trimmed_audio) / sample_rate
        print(f"  Sentence {idx}/{len(sentences)} done in {time.time() - st:.2f}s (duration: {chunk_dur:.2f}s)", flush=True)

    # 6. Concatenate and write WAV
    print("\n[4/5] Concatenating & saving final audio...", flush=True)
    final_audio = np.concatenate(audio_parts)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), final_audio, sample_rate, subtype="PCM_16")

    final_dur = len(final_audio) / sample_rate
    file_size_kb = out_file.stat().st_size / 1024

    print("\n[5/5] Verifying...", flush=True)
    print("==================================================", flush=True)
    print("=== CLONING FINISHED SUCCESSFULLY! ===")
    print(f"Output File : {out_file.resolve()}")
    print(f"Profile Pt  : {Path(profile_pt_path).resolve()}")
    print(f"Duration    : {final_dur:.2f} seconds")
    print(f"File Size   : {file_size_kb:.1f} KB")
    print(f"Sample Rate : {sample_rate} Hz (16-bit PCM WAV)")
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
