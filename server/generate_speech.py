import os
from datetime import datetime
import random
import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, TacotronSTFT
from tortoise.models.vocoder import UnivNetGenerator
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

class VoiceConditioningPipeline:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        # Singleton pattern to reuse the loaded models
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device=None):
        # Only initialize once
        if hasattr(self, 'tts'):
            return
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize TTS with optimizations
        self.tts = TextToSpeech(
            use_deepspeed=False,
            kv_cache=True,
            half=True,
            device=self.device,
            autoregressive_batch_size=16  # Increased batch size for faster processing
        )
        
        # Move models to GPU once and keep them there
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            self.tts.autoregressive = self.tts.autoregressive.to(self.device)
            self.tts.diffusion = self.tts.diffusion.to(self.device)
            self.tts.vocoder = self.tts.vocoder.to(self.device)
            self.tts.clvp = self.tts.clvp.to(self.device)
            
        # Initialize UnivNet vocoder for embeddings
        self.vocoder = UnivNetGenerator()
        if hasattr(UnivNetGenerator, 'get_pretrained'):
            self.vocoder = UnivNetGenerator.get_pretrained()
        self.vocoder = self.vocoder.to(self.device)
        self.vocoder.eval()
        
        # Initialize STFT for mel spectrogram computation
        self.stft = TacotronSTFT(
            1024, 256, 1024, 100, 24000, 0, 12000
        ).to(self.device)

    def generate_speech(self, text, voice_dir, output_path, preset="ultra_fast", num_autoregressive_samples=16, 
                       diffusion_iterations=30, cond_free=True):
        """Generate speech using voice latents."""
        try:
            # Load cached latents
            latents_path = Path(voice_dir) / "voice_latents.pth"
            if not latents_path.exists():
                raise ValueError("No cached latents found. Please run the caching script first.")
                
            # Load with torch.load using weights_only=True for faster loading
            latents = torch.load(latents_path, map_location=self.device)
            
            # Split text into chunks if it's too long
            chunks = self.split_text_into_chunks(text)
            if len(chunks) > 1:
                # Handle multiple chunks here if needed
                # For now, just use the first chunk
                text = chunks[0]
            
            # Generate speech with provided settings
            gen = self.tts.tts_with_preset(
                text=text,
                conditioning_latents=latents,
                preset=preset,
                num_autoregressive_samples=num_autoregressive_samples,
                diffusion_iterations=diffusion_iterations,
                cond_free=cond_free
            )
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save output
            if isinstance(gen, list):
                gen = gen[0]
            torchaudio.save(str(output_path), gen.squeeze(0).cpu(), 24000)
            
        except Exception as e:
            print(f"Error in generate_speech: {str(e)}")
            raise

    def split_text_into_chunks(self, text, chunk_size=300):
        """Split text into chunks of approximately chunk_size characters, preserving complete sentences and flow."""
        # Clean the text and normalize line endings
        text = ' '.join(text.split())
        
        # Split into sentences (considering multiple punctuation marks)
        sentences = []
        current_sentence = ""
        
        # Split by common sentence endings while preserving them
        for char in text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 0:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():  # Add any remaining text
            sentences.append(current_sentence.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)
            
        return chunks if chunks else [text]  # Return original text as single chunk if no splits needed

    def process_audio_file(self, input_path: str, output_path: str):
        """Process an audio file for voice conditioning."""
        try:
            print(f"Loading audio from: {input_path}")
            # Load and process the audio file
            audio = load_audio(input_path, 24000)
            
            # Convert to tensor if not already
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            
            # Ensure audio is 2D (channels x samples)
            if audio.dim() == 1:
                # If mono, add channel dimension
                audio = audio.unsqueeze(0)
            elif audio.dim() > 2:
                # If more than 2D, flatten to 2D
                audio = audio.view(audio.size(0), -1)
            
            # Ensure audio is on the correct device
            audio = audio.to(self.device)
            
            print(f"Saving processed audio to: {output_path}")
            # Save the processed audio (already 2D, no need for unsqueeze)
            torchaudio.save(output_path, audio.cpu(), 24000)
            
            # Verify the file was saved
            if not Path(output_path).exists():
                raise ValueError(f"Failed to save audio file to {output_path}")
                
            print(f"Successfully processed audio file: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            raise

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
            
            # Use our STFT instance directly
            mel = self.stft.mel_spectrogram(audio)
            mel = mel.to(self.device)
            
            # Generate random noise matching mel length
            z = torch.randn(1, self.vocoder.noise_dim, mel.size(2), device=self.device)
            
            with torch.no_grad():
                # Pass through initial convolution
                features = self.vocoder.conv_pre(z)
                
                # Get features from each res_stack block
                for res_block in self.vocoder.res_stack:
                    features = res_block(features, mel)
                
                # Extract fixed length features
                fixed_features = self.extract_fixed_length_features(features)
            
            return fixed_features.cpu().numpy()
            
        except Exception as e:
            print(f"Error in extract_features: {str(e)}")
            raise

    def save_embedding(self, voice_folder, embedding):
        """Save the voice embedding and metadata."""
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

    def cache_voice_latents(self, voice_dir: str):
        """Generate and cache voice latents for faster generation."""
        try:
            voice_dir = Path(voice_dir)
            if not voice_dir.exists():
                raise ValueError(f"Voice directory {voice_dir} does not exist")
                
            # Find all wav files in the directory
            wav_files = list(voice_dir.glob("*.wav"))
            print(f"Found {len(wav_files)} wav files in {voice_dir}")
            
            if not wav_files:
                raise ValueError(f"No wav files found in {voice_dir}")
            
            # Sort files by name and take first 20 chunks for latents
            wav_files = sorted(wav_files)[:20]
            print(f"Using first {len(wav_files)} chunks for voice conditioning")
            
            # Load and concatenate audio chunks
            all_audio = []
            embeddings = []
            for wav_path in wav_files:
                print(f"Loading wav file: {wav_path.name}")
                audio = load_audio(str(wav_path), 24000)
                
                # Convert to tensor if not already
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio)
                
                # Ensure audio is 1D
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)  # Convert to mono if needed
                    
                # Move to device
                audio = audio.to(self.device)
                
                # Add batch dimension if needed
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add batch dimension
                
                all_audio.append(audio)
                
                # Extract embedding features
                try:
                    embedding = self.extract_features(str(wav_path))
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Warning: Could not extract features from {wav_path.name}: {str(e)}")
            
            print("Generating voice latents...")
            # Generate voice latents using the list of audio tensors
            latents = self.tts.get_conditioning_latents(
                all_audio,  # List of properly shaped audio tensors
                return_mels=False
            )
            
            # Save latents
            latents_path = voice_dir / "voice_latents.pth"
            print(f"Saving voice latents to: {latents_path}")
            torch.save(latents, latents_path)
            
            if not latents_path.exists():
                raise ValueError(f"Failed to save voice latents to {latents_path}")
                
            # Process embeddings if we have any
            if embeddings:
                print("Processing voice embeddings...")
                embeddings = np.stack(embeddings)
                mean_embedding = np.mean(embeddings, axis=0)
                print(f"Created mean embedding with shape: {mean_embedding.shape}")
                
                # Save the embedding and metadata
                self.save_embedding(voice_dir, mean_embedding)
            else:
                print("Warning: Could not generate voice embeddings")
            
            print("Successfully cached voice latents and embeddings")
            return True
            
        except Exception as e:
            print(f"Error caching voice latents: {str(e)}")
            raise

def process_chunk_wrapper(args):
    chunk, voice_dir, out_path, device = args
    pipeline = VoiceConditioningPipeline(device=device)
    try:
        pipeline.generate_speech(chunk, voice_dir, out_path)
        return out_path, None
    except Exception as e:
        return out_path, str(e)

if __name__ == "__main__":
    # Generate a unique name for this generation
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    generation_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    generation_name = f"{current_time}_{generation_id}"
    
    # Create unique output directory for this generation
    output_base_dir = Path("output")
    generation_dir = output_base_dir / generation_name
    generation_dir.mkdir(parents=True, exist_ok=True)
    
    voice_dir = "./voices/voice4"
    text = """
    Once upon a time, in a land where the skies shimmered with colors unknown to ordinary eyes, there existed a magical forest known as Auroria. This forest was home to creatures of wonder—talking foxes, luminous butterflies, and even trees that hummed gentle melodies in the moonlight.

At the heart of Auroria was a Crystal Fountain, said to grant a single wish to anyone pure of heart. But the fountain was guarded by an ancient spell: only the truly selfless could unlock its power. For centuries, travelers tried and failed, blinded by greed or pride.

One day, a young girl named Lina ventured into Auroria. Lina wasn’t a princess or a warrior but a humble weaver who spun fabrics so fine they seemed to dance in the wind. Her village had been struck by a terrible drought, and her little brother, whom she adored, had fallen ill. Hearing tales of the Crystal Fountain, Lina set off to find it, hoping to wish for water to save her people.

The journey through Auroria was not easy. Lina encountered a raven with a broken wing trapped under a thorn bush. Though her time was precious, she stopped to bandage the bird and set it free. "Thank you," croaked the raven, its eyes twinkling. "Kindness is never forgotten."

Further along, Lina found a glade where an enormous bear was roaring in frustration. A beehive, its treasure out of reach, hung from a high branch. The bear growled, “Human, what do you seek here?” Despite her fear, Lina offered to help, crafting a long pole from nearby branches to gently retrieve the hive. The bear, now sated, bowed its massive head. “You have a generous heart,” it rumbled. “May fortune guide you.”

Finally, Lina arrived at the Crystal Fountain, its waters sparkling like liquid starlight. Before she could speak her wish, the fountain’s guardian appeared: a tall figure cloaked in silver mist. “Many come here seeking fortune or glory,” said the guardian. “What is your desire?”

“My people are thirsty, and my brother is sick,” Lina said. “I wish for water to restore life to my village.”

The guardian’s eyes gleamed. “And what of yourself? Would you not wish for wealth or a life of ease?”

Lina shook her head. “What joy would riches bring if those I love are gone?”

The guardian smiled and stepped aside. “You have proven yourself worthy.”

As Lina knelt to touch the water, it began to glow. When she returned to her village, a gentle rain followed her, soaking the parched earth. Her brother recovered, and the once-withered fields burst into bloom.

But that wasn’t all. From that day forward, Lina’s loom wove fabrics that sparkled with the colors of the Aurorian sky, bringing prosperity to her family and neighbors. And every night, a raven perched on her roof and a bear sat by her door, ensuring she was never alone.

And so, Lina lived a life filled with gratitude and joy, her selflessness forever etched into the heart of Auroria, where the Crystal Fountain still waits for those with pure hearts.
    """
    
    try:
        # Split text into chunks
        pipeline = VoiceConditioningPipeline()
        text_chunks = pipeline.split_text_into_chunks(text)
        print(f"\nGeneration ID: {generation_name}")
        print(f"Output directory: {generation_dir}")
        print(f"Split text into {len(text_chunks)} chunks")
        
        # Prepare arguments for parallel processing
        chunk_args = []
        output_files = []
        
        # If CUDA is available, determine number of GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        for i, chunk in enumerate(text_chunks):
            # Name format: part001.wav, part002.wav, etc.
            chunk_filename = f"part{i+1:03d}.wav"
            out_path = str(generation_dir / chunk_filename)
            output_files.append(out_path)
            
            # Assign GPU device if available (round-robin)
            device = f'cuda:{i % num_gpus}' if num_gpus > 0 else 'cpu'
            
            chunk_args.append((chunk, voice_dir, out_path, device))
        
        # Number of parallel workers (adjust based on available GPU memory)
        max_workers = min(num_gpus if num_gpus > 0 else 4, len(text_chunks))
        
        print(f"\nProcessing {len(text_chunks)} chunks using {max_workers} workers")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk_wrapper, args) for args in chunk_args]
            
            # Monitor progress
            for future in as_completed(futures):
                out_path, error = future.result()
                if error:
                    print(f"Error processing chunk {out_path}: {error}")
                else:
                    print(f"Successfully processed: {out_path}")
        
        print(f"\nAll chunks processed and saved in: {generation_dir}")
        print("Generated files (in order):")
        for f in sorted(output_files):
            print(f"- {Path(f).name}")
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()