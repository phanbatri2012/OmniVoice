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
    ref_audio_path = r"E:\yt\Tool\omnivoice\voice\SAM-1\SAM-1.WAV"
    output_path = r"E:\yt\Tool\omnivoice\output\sam_1_cloned.wav"
    temp_ref_crop_path = r"E:\yt\Tool\omnivoice\voice\SAM-1\SAM-1_ref_9s.wav"

    target_text = (
        "Có những người phụ nữ bị phản bội khi yếu đuối nhất, nhưng điều đáng sợ không phải là bị bỏ rơi mà là phát hiện mọi thứ đã được tính toán từ trước. "
        "Đêm ấy, Linh vừa sinh con chưa đầy tháng."
    )

    # 1. Prepare 9.2s reference audio & exact transcript (within 3-10s standard)
    ref_text = "tôi muốn mời quý vị nghe câu chuyện của một người phụ nữ bị đẩy ra khỏi chính căn nhà mình từng góp sức xây nên vào lúc cơ thể cô còn chưa hồi phục sau sinh."
    
    # Read and crop reference audio to 0.0s - 9.2s
    raw_ref, ref_sr = sf.read(ref_audio_path, dtype="float32", always_2d=True)
    crop_samples = int(9.2 * ref_sr)
    mono_ref = raw_ref[:crop_samples].mean(axis=1)
    sf.write(temp_ref_crop_path, mono_ref, ref_sr, subtype="PCM_16")

    print("=== [OmniVoice Voice Clone - SAM-1] ===")
    print(f"Ref Audio (Cropped 9.2s): {temp_ref_crop_path}")
    print(f"Ref Text: {ref_text}\n")

    # 2. Strict sentence chunking by punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', target_text.strip()) if s.strip()]
    print(f"Target Sentences ({len(sentences)}):")
    for idx, s in enumerate(sentences, 1):
        print(f"  [{idx}] {s}")

    # 3. Model & Generation Config
    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        denoise=False,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    print("\n[1/4] Loading OmniVoice model on CUDA float16...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
    )
    sample_rate = int(model.sampling_rate)
    print(f"Loaded model in {time.time() - t0:.2f}s | Sample rate: {sample_rate} Hz")

    # 4. Create Voice Clone Prompt
    print("\n[2/4] Extracting voice prompt from reference audio & text...")
    t1 = time.time()
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=temp_ref_crop_path,
        ref_text=ref_text,
        preprocess_prompt=True,
    )
    print(f"Voice prompt extracted in {time.time() - t1:.2f}s")

    # 5. Generate audio chunk by chunk
    print("\n[3/4] Synthesizing sentences...")
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
        print(f"  Sentence {idx}/{len(sentences)} done in {time.time() - st:.2f}s (duration: {chunk_dur:.2f}s)")

    # 6. Concatenate and write WAV
    print("\n[4/4] Concatenating & saving final audio...")
    final_audio = np.concatenate(audio_parts)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), final_audio, sample_rate, subtype="PCM_16")

    final_dur = len(final_audio) / sample_rate
    file_size_kb = out_file.stat().st_size / 1024

    print("\n==================================================")
    print("=== CLONING FINISHED SUCCESSFULLY! ===")
    print(f"Output File : {out_file.resolve()}")
    print(f"Duration    : {final_dur:.2f} seconds")
    print(f"File Size   : {file_size_kb:.1f} KB")
    print(f"Sample Rate : {sample_rate} Hz (16-bit PCM WAV)")
    print("==================================================")

if __name__ == "__main__":
    main()
