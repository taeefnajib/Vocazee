from fastapi import FastAPI, HTTPException, UploadFile, Body, Form, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import time
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import string

from generate_speech import VoiceConditioningPipeline

app = FastAPI(title="Vocazee API", description="Voice Generation and Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Vite's default dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
BASE_DIR = Path(__file__).parent
Path(BASE_DIR / "output").mkdir(exist_ok=True)
Path(BASE_DIR / "train_voices").mkdir(exist_ok=True)
Path(BASE_DIR / "voices").mkdir(exist_ok=True)

# Initialize pipeline lazily
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Initializing voice conditioning pipeline...")
        pipeline = VoiceConditioningPipeline()
    return pipeline

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Vocazee API server...")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info("Creating required directories...")
    
    # Create output directory if it doesn't exist
    Path(BASE_DIR / "output").mkdir(exist_ok=True)
    Path(BASE_DIR / "train_voices").mkdir(exist_ok=True)
    Path(BASE_DIR / "voices").mkdir(exist_ok=True)
    
    logger.info("Server initialization complete!")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Vocazee API is running"}

def convert_to_wav(input_path: Path) -> Path:
    """Convert audio file to WAV format if needed."""
    if input_path.suffix.lower() == '.wav':
        return input_path
        
    output_path = input_path.with_suffix('.wav')
    audio = AudioSegment.from_file(str(input_path))
    audio.export(str(output_path), format='wav')
    return output_path

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

@app.post("/generate-speech")
async def generate_speech(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        voice_name = data.get("voice_name", "")
        high_quality = data.get("high_quality", False)
        
        if not text or not voice_name:
            raise HTTPException(status_code=400, detail="Missing text or voice name")
            
        voice_dir = BASE_DIR / "voices" / voice_name
        if not voice_dir.exists():
            raise HTTPException(status_code=404, detail=f"Voice {voice_name} not found")
            
        # Generate a unique ID for this generation
        generation_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        output_dir = BASE_DIR / "output" / generation_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split text into chunks and generate audio for each
        chunks = get_pipeline().split_text_into_chunks(text)
        for i, chunk in enumerate(chunks):
            output_path = output_dir / f"part{str(i + 1).zfill(3)}.wav"
            
            # Use different quality settings based on the toggle
            if high_quality:
                get_pipeline().generate_speech(
                    text=chunk,
                    voice_dir=str(voice_dir),
                    output_path=str(output_path),
                    preset="high_quality",
                    num_autoregressive_samples=64,
                    diffusion_iterations=120,
                    cond_free=True
                )
            else:
                get_pipeline().generate_speech(
                    text=chunk,
                    voice_dir=str(voice_dir),
                    output_path=str(output_path),
                    preset="ultra_fast",
                    num_autoregressive_samples=16,
                    diffusion_iterations=30,
                    cond_free=True
                )
        
        return {
            "status": "success",
            "generation_id": generation_id,
            "num_chunks": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-voice")
async def create_voice(
    voice_name: str = Form(...),
    file: UploadFile = File(...)
):
    voice_dir = None
    output_dir = None
    temp_file = None
    try:
        # Check if file extension is supported
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.wav', '.mp3', '.webm']:
            raise ValueError("Only .wav, .mp3, and .webm files are supported")
            
        # Create voice directories
        voice_dir = BASE_DIR / "train_voices" / voice_name
        output_dir = BASE_DIR / "voices" / voice_name
        voice_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Save uploaded file to train_voices directory
        temp_path = voice_dir / file.filename
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Convert to WAV and load audio
        print(f"Converting {temp_path} to WAV format...")
        audio = AudioSegment.from_file(str(temp_path))
        if temp_path.suffix.lower() != '.wav':
            temp_file = temp_path  # Store original file path for cleanup
            
        # Remove silence and get chunks
        print("Removing silence...")
        chunks = remove_silence(audio)
        
        # Combine short chunks to meet length requirements
        print("Combining chunks to meet length requirements...")
        processed_chunks = combine_short_chunks(chunks)
        
        print(f"Processing {len(processed_chunks)} audio chunks...")
        for i, chunk in enumerate(processed_chunks):
            # Save chunk to WAV file
            chunk_path = output_dir / f"chunk_{i:04d}.wav"
            chunk.export(str(chunk_path), format="wav")
            
            # Process the chunk for voice conditioning
            success = get_pipeline().process_audio_file(str(chunk_path), str(chunk_path))
            if not success:
                raise ValueError(f"Failed to process audio chunk {i}")
                
        # Generate voice latents after processing all chunks
        print(f"Generating voice latents for: {output_dir}")
        success = get_pipeline().cache_voice_latents(str(output_dir))
        if not success:
            raise ValueError("Failed to generate voice latents")
        
        # Clean up temporary files
        if temp_file and temp_file.exists():
            temp_file.unlink()
        
        return {
            "status": "success", 
            "voice_name": voice_name,
            "num_chunks": len(processed_chunks)
        }
        
    except Exception as e:
        print(f"Error in create_voice: {str(e)}")
        # Clean up temporary files
        if temp_file and temp_file.exists():
            temp_file.unlink()
        # Only clean up if we failed to process the voice completely
        if voice_dir and voice_dir.exists():
            shutil.rmtree(voice_dir)
        if output_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """List all available voice profiles."""
    try:
        voices_dir = BASE_DIR / "voices"
        if not voices_dir.exists():
            return {"voices": []}
            
        # Get all directories in voices folder
        voices = [
            dir.name for dir in voices_dir.iterdir() 
            if dir.is_dir() and (dir / "voice_latents.pth").exists()
        ]
        
        return {
            "status": "success",
            "voices": sorted(voices)  # Sort alphabetically
        }
        
    except Exception as e:
        print(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{generation_id}/{part}")
async def get_audio(generation_id: str, part: str):
    try:
        # Look in the output directory instead of output
        audio_path = BASE_DIR / "output" / generation_id / part
        print(f"Looking for audio file at: {audio_path}")
        
        if not audio_path.exists():
            print(f"File not found at: {audio_path}")
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        return FileResponse(str(audio_path), media_type="audio/wav")
        
    except Exception as e:
        print(f"Error in get_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
