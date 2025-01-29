import torch
import torchaudio
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from tortoise.utils.audio import load_audio, load_voice, load_voices, wav_to_univnet_mel, TacotronSTFT
from tortoise.models.vocoder import UnivNetGenerator

class VoiceEmbeddingCache:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize UnivNet vocoder
        self.vocoder = UnivNetGenerator()
        if hasattr(UnivNetGenerator, 'get_pretrained'):
            self.vocoder = UnivNetGenerator.get_pretrained()
        self.vocoder = self.vocoder.to(self.device)
        self.vocoder.eval()
        
        # Initialize STFT for mel spectrogram computation
        self.stft = TacotronSTFT(
            1024, 256, 1024, 100, 24000, 0, 12000
        ).to(self.device)

    def extract_fixed_length_features(self, features, target_len=100):
        """Extract fixed length features using adaptive pooling."""
        # Use adaptive pooling to get fixed length
        pool = torch.nn.AdaptiveAvgPool1d(target_len)
        
        # Apply pooling to get fixed length features
        fixed_features = pool(features)
        
        # Get channel statistics
        mean_features = torch.mean(fixed_features, dim=2)  # Mean across time
        std_features = torch.std(fixed_features, dim=2)    # Std across time
        
        # Concatenate statistics
        combined_features = torch.cat([mean_features, std_features], dim=1)
        return combined_features

    def extract_features(self, audio_path):
        """Extract features from audio using UnivNet's internal representations."""
        try:
            # Load audio using Tortoise's utility
            audio = load_audio(audio_path, 24000)  # Using 24kHz to match STFT config
            
            # Move audio to device and ensure it's the right shape
            audio = audio.to(self.device)
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            print(f"Audio shape: {audio.shape}, device: {audio.device}")
            
            # Use our STFT instance directly
            mel = self.stft.mel_spectrogram(audio)
            mel = mel.to(self.device)
            print(f"Mel shape: {mel.shape}, device: {mel.device}")
            
            # Generate random noise matching mel length
            z = torch.randn(1, self.vocoder.noise_dim, mel.size(2), device=self.device)
            print(f"Noise shape: {z.shape}, device: {z.device}")
            
            with torch.no_grad():
                # Pass through initial convolution
                features = self.vocoder.conv_pre(z)
                print(f"Initial features shape: {features.shape}, device: {features.device}")
                
                # Get features from each res_stack block
                for i, res_block in enumerate(self.vocoder.res_stack):
                    features = res_block(features, mel)
                    print(f"Features after res_block {i} shape: {features.shape}, device: {features.device}")
                
                # Extract fixed length features
                fixed_features = self.extract_fixed_length_features(features)
                print(f"Fixed length features shape: {fixed_features.shape}, device: {fixed_features.device}")
            
            return fixed_features.cpu().numpy()
            
        except Exception as e:
            print(f"Detailed error in extract_features: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def process_voice_folder(self, voice_folder):
        """Process all audio files in a voice folder and compute embeddings."""
        voice_folder = Path(voice_folder)
        embeddings = []
        
        # Process each audio file in the voice folder
        audio_files = list(voice_folder.glob("*.wav"))
        if not audio_files:
            raise ValueError(f"No .wav files found in {voice_folder}")
        
        print(f"Found {len(audio_files)} WAV files in {voice_folder}")
            
        for audio_file in audio_files:
            try:
                print(f"\nProcessing {audio_file}...")
                embedding = self.extract_features(str(audio_file))
                embeddings.append(embedding)
                print(f"Successfully processed {audio_file}")
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {str(e)}")
                continue
        
        if not embeddings:
            raise ValueError(f"Could not extract features from any audio files in {voice_folder}")
        
        # Stack embeddings and compute mean
        embeddings = np.stack(embeddings)
        mean_embedding = np.mean(embeddings, axis=0)
        print(f"Created mean embedding with shape: {mean_embedding.shape}")
        
        # Save the computed embedding in the voice folder
        self.save_embedding(voice_folder, mean_embedding)
        
        return mean_embedding

    def save_embedding(self, voice_folder, embedding):
        """Save the voice embedding directly in the voice folder."""
        voice_folder = Path(voice_folder)
        embedding_path = voice_folder / "voice_embedding.npy"
        np.save(embedding_path, embedding)
        
        # Save metadata
        metadata = {
            "voice_name": voice_folder.name,
            "creation_date": str(datetime.now()),
            "embedding_shape": list(embedding.shape),
            "model": "univnet",
            "version": "1.0"
        }
        
        metadata_path = voice_folder / "voice_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved embedding to {embedding_path}")
        print(f"Saved metadata to {metadata_path}")

    def load_embedding(self, voice_folder):
        """Load a pre-computed voice embedding from the voice folder."""
        voice_folder = Path(voice_folder)
        embedding_path = voice_folder / "voice_embedding.npy"
        
        if embedding_path.exists():
            embedding = np.load(embedding_path)
            return torch.from_numpy(embedding).to(self.device)
        else:
            raise FileNotFoundError(f"No cached embedding found in {voice_folder}")

    def get_or_compute_embedding(self, voice_folder):
        """Get cached embedding or compute if not available."""
        try:
            return self.load_embedding(voice_folder)
        except FileNotFoundError:
            print(f"Computing new embedding for {Path(voice_folder).name}")
            return self.process_voice_folder(voice_folder)

def main():
    # Initialize the cache
    cache = VoiceEmbeddingCache()
    
    # Example usage with a single voice folder
    voice_folder = "./voices/Ana"
    try:
        embedding = cache.get_or_compute_embedding(voice_folder)
        print(f"Successfully processed voice: {Path(voice_folder).name}")
        print(f"Embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"Error processing voice: {e}")

if __name__ == "__main__":
    main()