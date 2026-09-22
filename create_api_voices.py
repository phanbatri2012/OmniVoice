import torch
import soundfile as sf
import os
from pathlib import Path
from omnivoice.models.omnivoice import OmniVoice

print("Loading OmniVoice model...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", 
    device_map="cuda", 
    dtype=torch.float16
)

# Thư mục lưu Voice Profile của API
profile_dir = Path("E:/yt/Tool/omnivoice/voice_library")
profile_dir.mkdir(parents=True, exist_ok=True)

voices_to_create = [
    {
        "name": "dinh_doan_chuan",
        "audio_path": "E:/yt/Tool/omnivoice/voice/dinh_doan/dinh_doan/dinh_doan_14s.WAV",
        "ref_text": "Xin chào tất cả các bác a những người đã trên 60 gọi là là người gọi là người có tuổi à tôi đã rất là kiên trì đồng hành với tất cả các bác các anh các chị những người có tuổi trong một thời gian rất là dài vừa rồi"
    },
    {
        "name": "cd_gkvs_chuan",
        "audio_path": "E:/yt/Tool/omnivoice/voice/CD_GKVS/GKVS/GKVS_14s.WAV",
        "ref_text": "30/4 năm một chín 7 5 cuộc kháng chiến chống Mỹ cứu nước kết thúc thắng lợi Bắc Nam sum họp một nhà non sông nối liền một dải nhưng tưởng giờ đây Việt Nam đã có được hòa bình và ổn định để khắc phục hậu quả chiến tranh"
    }
]

for voice in voices_to_create:
    print(f"Creating profile for {voice['name']}...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=voice["audio_path"],
        ref_text=voice["ref_text"]
    )
    
    out_path = profile_dir / f"{voice['name']}.pt"
    prompt.save(str(out_path))
    print(f"-> Saved profile to {out_path}")

print("Done! Bạn có thể sử dụng 2 giọng này qua API rồi.")
