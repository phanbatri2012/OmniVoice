# OmniVoice API & Script Guidelines

When working with OmniVoice generation (both in Python scripts and `api_server.py`) in this project, **always adhere to the following configurations** to ensure high-quality, hallucination-free Vietnamese TTS:

1. **Ref Text is Mandatory for Voice Cloning**:
   - Always supply `ref_text` along with `ref_audio` to `model.generate()` or `model.create_voice_clone_prompt()`.
   - Never clear or set `ref_text=""` during inference, as this causes the Whisper model to hallucinate (e.g., "Ghiền Mì Gõ") and severely degrades voice timing and clone accuracy.

2. **Strict Sentence Chunking**:
   - Do NOT split long text purely by character count (e.g., 220 chars).
   - Use regex to split text strictly by punctuation (e.g. `re.split(r'(?<=[.!?\n])\s+', text)`) so that each chunk is exactly one natural sentence.

3. **Inference Quality Configurations**:
   - `num_step`: Always set to **32** (defaulting to 16 causes degraded "sandy" audio).
   - `duration` / `target_duration`: Set to **None** (do NOT force a calculated duration based on words-per-minute, as it stretches or compresses the audio unnaturally, causing distortion).

4. **Audio Chunk Merging (Pop/Click Prevention)**:
   - When manually concatenating audio chunks or inserting silence using `numpy`, **always** apply a short crossfade or envelope (e.g., a 10ms fade-in/fade-out using `np.linspace(0.0, 1.0, fade_samples)`).
   - Directly appending `np.zeros()` to raw model output without fading the boundaries to 0.0 creates abrupt amplitude jumps, causing audible "pop" or "click" artifacts between sentences.
