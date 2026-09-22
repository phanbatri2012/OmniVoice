import os
import glob
import logging
from typing import Optional, Dict
from omnivoice.models.omnivoice import VoiceClonePrompt

logger = logging.getLogger(__name__)

class VoiceLibrary:
    """
    Quản lý các profile giọng nói (được lưu dưới dạng file .pt chứa prompt đã encode)
    """
    def __init__(self, library_dir: str = "voices"):
        self.library_dir = library_dir
        self.voices: Dict[str, VoiceClonePrompt] = {}
        self._ensure_library_dir()
        self.load_all()

    def _ensure_library_dir(self):
        if not os.path.exists(self.library_dir):
            os.makedirs(self.library_dir)
            logger.info(f"Created voice library directory at: {self.library_dir}")

    def load_all(self):
        """Tải tất cả các file .pt trong thư mục vào bộ nhớ"""
        self.voices.clear()
        pattern = os.path.join(self.library_dir, "*.pt")
        for pt_file in glob.glob(pattern):
            try:
                voice_name = os.path.basename(pt_file)[:-3] # Bỏ đuôi .pt
                prompt = VoiceClonePrompt.load(pt_file)
                self.voices[voice_name] = prompt
                logger.info(f"Loaded voice prompt: {voice_name}")
            except Exception as e:
                logger.error(f"Failed to load voice {pt_file}: {e}")

    def get_voice(self, voice_name: str) -> Optional[VoiceClonePrompt]:
        """Lấy một prompt theo tên (không cần .pt)"""
        return self.voices.get(voice_name)

    def add_voice_from_prompt(self, voice_name: str, prompt: VoiceClonePrompt):
        """Thêm và lưu một prompt mới"""
        save_path = os.path.join(self.library_dir, f"{voice_name}.pt")
        prompt.save(save_path)
        self.voices[voice_name] = prompt
        logger.info(f"Saved new voice prompt: {voice_name} to {save_path}")

    def list_voices(self):
        return list(self.voices.keys())

# Khởi tạo singleton mặc định
default_voice_library = VoiceLibrary(library_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_library"))
