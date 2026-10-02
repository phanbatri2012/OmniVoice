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

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

# 1. Paths & Inputs
REF_AUDIO_PATH = r"E:\yt\Tool\omnivoice\voice\nu_tre_1\nu_tre_1.WAV"
REF_SRT_PATH = r"E:\yt\Tool\omnivoice\voice\nu_tre_1\nu_tre_1.srt"
OUTPUT_PATH = r"E:\yt\Tool\omnivoice\output\nu_tre_1_cloned.wav"

TARGET_TEXT = (
    "Cháu vẫn nhớ rất rõ cái lạnh ngoài cổng hôm đó. "
    "Một tay cháu ôm con, một tay kéo chiếc vali Khải đã chuẩn bị sẵn. "
    "Trong nhà Vy người bạn thân hơn mười năm của cháu. "
    "Đứng cạnh chồng cháu như thể cô ấy mới là chủ nhân căn nhà."
)

# Extract ref_text from SRT
def extract_srt_text(srt_path: str) -> str:
    content = Path(srt_path).read_text(encoding="utf-8")
    lines = content.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        text_lines.append(line)
    return ". ".join(text_lines) + "."

ref_text = extract_srt_text(REF_SRT_PATH)
print("=== REF TEXT EXTRACTED ===")
print(ref_text)
print("==========================")

# 2. Split Target Text by punctuation
sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', TARGET_TEXT.strip()) if s.strip()]
print(f"Target sentences ({len(sentences)}):")
for idx, s in enumerate(sentences, 1):
    print(f"  [{idx}] {s}")

# 3. Model Configuration
gen_config = OmniVoiceGenerationConfig(
    num_step=32,
    denoise=False,
    preprocess_prompt=True,
    postprocess_output=True,
)

print("\nLoading OmniVoice model on CUDA float16...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda",
    dtype=torch.float16,
)

sample_rate = model.sampling_rate
print(f"Model loaded successfully. Sample rate: {sample_rate} Hz")

# 4. Create Voice Clone Prompt (Cached for fast & consistent generation)
print("Creating voice clone prompt with ref_audio & ref_text...")
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO_PATH,
    ref_text=ref_text,
    preprocess_prompt=True,
)

# 5. Generate Audio for each sentence
audio_parts = []
pause_seconds = 0.25  # 250ms natural pause between sentences
fade_samples = int(sample_rate * 0.01)  # 10ms crossfade

for idx, sentence in enumerate(sentences, 1):
    print(f"\n-> Generating sentence {idx}/{len(sentences)}: '{sentence}'")
    start_t = time.time()
    
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
        
    # Trim extreme silence at boundaries
    trimmed_audio, _ = librosa.effects.trim(raw_audio, top_db=40)
    if trimmed_audio.size == 0:
        trimmed_audio = raw_audio

    # Apply 10ms fade-in and fade-out to prevent pops/clicks
    if len(trimmed_audio) > fade_samples * 2:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed_audio[:fade_samples] *= fade_in
        trimmed_audio[-fade_samples:] *= fade_out

    audio_parts.append(trimmed_audio)
    
    # Add silence between sentences (except after the last sentence)
    if idx < len(sentences):
        silence = np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
        audio_parts.append(silence)
        
    dur = len(trimmed_audio) / sample_rate
    elapsed = time.time() - start_t
    print(f"   Done in {elapsed:.2f}s | Audio duration: {dur:.2f}s")

# 6. Concatenate & Save
print("\nConcatenating audio parts...")
final_audio = np.concatenate(audio_parts)

output_file = Path(OUTPUT_PATH)
output_file.parent.mkdir(parents=True, exist_ok=True)

sf.write(str(output_file), final_audio, sample_rate, subtype="PCM_16")
final_duration = len(final_audio) / sample_rate
msg = f"SUCCESS! Cloned audio saved to: {output_file.resolve()}\nTotal duration: {final_duration:.2f}s | Sample rate: {sample_rate} Hz | Format: PCM_16 WAV"
print(msg)
with open(r"E:\yt\Tool\omnivoice\clone.log", "w", encoding="utf-8") as f:
    f.write(msg + "\n")

