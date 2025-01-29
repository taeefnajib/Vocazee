import os
import torch
import warnings
from pathlib import Path
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio

warnings.filterwarnings("ignore")

def cache_voice_latents(voice_dir):
    """Cache voice latents for faster generation later."""
    voice_dir = Path(voice_dir)
    latents_path = voice_dir / "voice_latents.pth"
    
    if latents_path.exists():
        print("Latents already cached.")
        return
        
    # Initialize TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
    
    # Load voice samples
    print("Processing voice samples...")
    voice_samples = []
    for ext in ['*.wav', '*.mp3']:
        voice_samples.extend([load_audio(str(p), 22050) for p in voice_dir.glob(ext)])
        
    if not voice_samples:
        raise ValueError(f"No audio files found in {voice_dir}")
        
    # Get conditioning latents
    print("Generating conditioning latents...")
    conditioning_latents = tts.get_conditioning_latents(voice_samples)
    
    # Cache the latents
    print("Saving latents...")
    torch.save(conditioning_latents, latents_path)
    print(f"Saved latents to {latents_path}")

if __name__ == "__main__":
    voice_dir = "./voices/Ana"
    cache_voice_latents(voice_dir)