"""Audio processing utilities for voice conditioning."""
import torchaudio
import torch
import numpy as np
from pathlib import Path

def process_audio_file(input_path, output_path, target_sr=24000):
    """Process an audio file for voice conditioning.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save processed audio
        target_sr (int): Target sample rate (default: 24000 for Tortoise)
    """
    print(f"Processing audio file: {input_path}")
    print(f"Output path: {output_path}")
    
    try:
        # Load audio
        audio, sr = torchaudio.load(input_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
        
        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))
        
        # Save processed audio
        torchaudio.save(output_path, audio, target_sr)
        print(f"Successfully processed audio to: {output_path}")
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise
