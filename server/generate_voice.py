import os
import sys
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import webrtcvad
import wave
import array
import contextlib

def convert_to_wav(input_file):
    """Convert input audio to wav format if needed."""
    audio = AudioSegment.from_file(input_file)
    return audio

def remove_silence(audio, min_silence_len=500, silence_thresh=-40):
    """Remove silence from the audio."""
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100  # Keep 100ms of silence at the edges
    )
    return chunks

def combine_short_chunks(chunks, min_length=10000, max_length=15000):
    """Combine short audio chunks to meet the desired length requirements."""
    combined_chunks = []
    current_chunk = None
    
    for chunk in chunks:
        if current_chunk is None:
            current_chunk = chunk
        else:
            if len(current_chunk) + len(chunk) <= max_length:
                current_chunk += chunk
            else:
                if len(current_chunk) >= min_length:
                    combined_chunks.append(current_chunk)
                current_chunk = chunk
    
    if current_chunk is not None and len(current_chunk) >= min_length:
        combined_chunks.append(current_chunk)
    
    return combined_chunks

def process_audio_file(voice_name, input_file, chunk_offset=0):
    """Process a single audio file and return the number of chunks created."""
    # Create output directory
    output_dir = Path(f"voices/{voice_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to wav and load audio
    print(f"\nProcessing audio file: {input_file}")
    audio = convert_to_wav(input_file)
    
    # Remove silence
    print("Removing silence...")
    chunks = remove_silence(audio)
    
    # Combine short chunks to meet length requirements
    print("Combining chunks to meet length requirements...")
    processed_chunks = combine_short_chunks(chunks)
    
    # Save chunks
    print(f"Saving {len(processed_chunks)} audio chunks...")
    for i, chunk in enumerate(processed_chunks, start=chunk_offset):
        output_file = output_dir / f"chunk_{i:04d}.wav"
        chunk.export(str(output_file), format="wav")
    
    return len(processed_chunks)

def process_voice_directory(voice_name, input_dir):
    """Process all audio files in a directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")
    
    # Get all audio files
    audio_files = []
    for ext in ['.wav', '.mp3']:
        audio_files.extend(input_path.glob(f"*{ext}"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in '{input_dir}'")
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Process each file
    chunk_offset = 0
    for audio_file in sorted(audio_files):
        chunks_created = process_audio_file(voice_name, audio_file, chunk_offset)
        chunk_offset += chunks_created
    
    print(f"\nProcessing complete. All audio chunks saved in: voices/{voice_name}")
    print(f"Total chunks created: {chunk_offset}")

# Set your voice name and input directory here
voice_name = "Ana"  # Change this to your desired voice name
input_directory = "./train_voices/Ana"  # Directory containing audio files

if __name__ == "__main__":
    try:
        process_voice_directory(voice_name, input_directory)
    except Exception as e:
        print(f"Error processing voice: {e}")