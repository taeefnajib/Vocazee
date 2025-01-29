# Vocazee

A voice cloning and text-to-speech application that can generate speech in any voice.

## Technical Details

- Frontend: React
- Backend: FastAPI
- Text-to-speech: Tortoise TTS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/taeefnajib/vocazee.git
cd vocazee
```

2. Build and start the containers:
```bash
docker compose up --build
```
Note: The first build will take some time as it downloads necessary AI models (>1GB). This is a one-time setup.

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

### Training a Custom Voice

1. From the Web Interface:
   - Go to http://localhost:3000
   - Switch to the "Train Custom Voice" tab
   - Enter a name for your voice
   - Record a clear audio file of your voice reading the provided text
   - Click "Train Voice"
   - Wait for the training to complete (usually takes 1-2 minutes)

2. From Command Line (Advance):
   Go to `server` directory. Make sure you create and activate a virtual environment and install dependencies by running `pip install -r requirements.txt`. Now follow the steps:
   ```bash
   # 1. First, process your audio file
   python generate_voice.py --input_file path/to/your/audio.wav --output_dir voices/your_voice_name

   # 2. Generate voice embeddings
   python save_embeddings.py --voice_dir voices/your_voice_name

   # 3. Cache voice latents for faster generation
   python cache_voice_latents.py --voice_dir voices/your_voice_name
   ```

Tips for best results:
- Use high-quality audio with minimal background noise
- Record in a quiet environment
- Speak clearly and at a natural pace
- Aim for at least 120 seconds of audio

### Generating Speech

1. From the Web Interface:
   - Go to http://localhost:3000
   - Select a trained voice from the dropdown
   - Enter or paste the text you want to convert to speech
   - Toggle "High Quality" if desired (slower but better quality)
   - Click "Generate Speech"
   - Once complete, use the audio player to listen or download the generated audio

2. Using the API directly:
   ```bash
   curl -X POST http://localhost:8000/generate-speech \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your text here",
       "voice_name": "your_voice_name",
       "high_quality": false
     }'
   ```

## Voice Directory Structure

Each voice in the `voices` directory should have the following structure:
```
voices/
└── your_voice_name/
    ├── original.wav           # Original audio file
    ├── chunks               # Processed audio chunks
    ├── voice_latents.pth    # Cached voice latents
    └── embeddings.pt        # Voice embeddings
```

## API Endpoints

- `POST /create-voice`: Train a new voice
- `GET /voices`: List all available voices
- `POST /generate-speech`: Generate speech from text
- `GET /audio/{generation_id}/{part}`: Get generated audio file

## Troubleshooting

1. If the server is slow on first request:
   - This is normal as models are being loaded
   - Subsequent requests will be faster

2. If voice training fails:
   - Ensure audio is clear and has minimal background noise
   - Try recording a longer sample
   - Check if the audio format is supported (WAV recommended)

3. If speech generation is stuck:
   - Check server logs using `docker logs vocazee-server-1`
   - Ensure the voice model exists and is properly trained
   - Try with a shorter text first

## License

MIT License
