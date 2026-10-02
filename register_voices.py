import sys
import os
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import soundfile as sf
from omnivoice.models.omnivoice import OmniVoice

print("=== [Registering Voices into voice_library] ===")
profile_dir = Path(r"E:\yt\Tool\omnivoice\voice_library")
profile_dir.mkdir(parents=True, exist_ok=True)

voices_to_register = [
    {
        "name": "nu_tre_1",
        "audio_path": r"E:\yt\Tool\omnivoice\voice\nu_tre_1\nu_tre_1_ref_7s.wav",
        "ref_text": "cháu vẫn nhớ rất rõ cái lạnh ngoài cổng hôm đó, một tay cháu ôm con, một tay kéo chiếc vali Khải đã chuẩn bị sẵn."
    },
    {
        "name": "sam_1",
        "audio_path": r"E:\yt\Tool\omnivoice\voice\SAM-1\SAM-1_ref_9s.wav",
        "ref_text": "tôi muốn mời quý vị nghe câu chuyện của một người phụ nữ bị đẩy ra khỏi chính căn nhà mình từng góp sức xây nên vào lúc cơ thể cô còn chưa hồi phục sau sinh."
    }
]

print("Loading OmniVoice model on CUDA...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", 
    device_map="cuda", 
    dtype=torch.float16
)

for v in voices_to_register:
    print(f"\n-> Creating voice prompt for '{v['name']}'...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=v["audio_path"],
        ref_text=v["ref_text"],
        preprocess_prompt=True
    )
    out_pt = profile_dir / f"{v['name']}.pt"
    prompt.save(str(out_pt))
    print(f"   Saved profile prompt to: {out_pt} ({out_pt.stat().st_size} bytes)")

print("\n=== SUCCESS: All voices registered to voice_library! ===")
