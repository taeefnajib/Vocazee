import React, { useState, useRef, useEffect } from 'react';
import { Mic, Play, Volume2, StopCircle, RefreshCw } from 'lucide-react';
import { createVoiceProfile, generateSpeech, getAudioUrl, getVoices } from './api';

const TRAINING_TEXT_1 = `Once upon a time, there were three little pigs who decided it was time to leave their mother’s house and make their own way in the world. Each pig wanted to build a house, but they had different ideas about how to do it. The first pig, eager to relax and enjoy life, quickly built his house out of straw. "This was so easy!" he said, admiring his flimsy creation. The second pig, wanting a house with more durability but still not keen on hard work, chose sticks. "My house is stronger than yours," he boasted to the first pig, though it wasn’t much better. The third pig, the wisest of the three, decided to work hard and build his house from bricks. He labored for days, laying each brick carefully, and soon had a sturdy and safe home.

One day, a hungry wolf came upon the first pig’s straw house. Knocking on the door, he said, "Little pig, little pig, let me come in." The pig replied, "Not by the hair on my chinny-chin-chin!" So, the wolf huffed, and he puffed, and he blew the straw house down. The terrified pig ran to his brother’s stick house for safety.

The wolf followed and knocked on the stick house door. "Little pigs, little pigs, let me come in!" he called. "Not by the hair on our chinny-chin-chins!" they shouted back. So, the wolf huffed, and he puffed, and he blew the stick house down, sending the two little pigs scrambling to their brother’s brick house.

When the wolf arrived at the brick house, he knocked and repeated his demand. "Little pigs, little pigs, let me come in!" But the three pigs answered confidently, "Not by the hair on our chinny-chin-chins!" The wolf huffed, and he puffed, but no matter how hard he tried, the brick house stood firm. Exhausted but determined, the wolf climbed onto the roof, planning to sneak down the chimney.

The clever third pig, hearing the wolf’s plan, quickly lit a roaring fire in the fireplace and placed a large pot of boiling water beneath the chimney. As the wolf slid down, he landed straight in the pot with a yelp and leapt out, fleeing into the woods, never to bother the pigs again. From that day on, the three little pigs lived happily and safely in the sturdy brick house, grateful for their brother’s wisdom and hard work. They learned the importance of doing things properly and working hard to ensure lasting success.`;

const TRAINING_TEXT_2 = `Once upon a time, in a lush green forest, there lived a young girl named Goldilocks. One sunny morning, she wandered away from home, exploring the woods, and soon came across a quaint little cottage. Curious, she knocked on the door, but when no one answered, she decided to go inside. The door creaked open, revealing a cozy room with three bowls of porridge on the table. Hungry from her walk, Goldilocks tasted the first bowl. "Too hot!" she exclaimed. She tried the second bowl. "Too cold!" she said. Finally, she tasted the third bowl. "Just right!" she smiled and ate it all.

Afterward, she noticed three chairs by the fireplace. She sat in the first chair. "Too hard!" she muttered. She tried the second chair. "Too soft!" she said. Then she sat in the third chair. "Just right!" she sighed, but the chair couldn’t hold her weight and broke into pieces. Feeling tired, Goldilocks climbed the stairs and found three beds. She lay on the first bed. "Too hard!" she complained. She tried the second bed. "Too soft!" she murmured. Finally, she lay on the third bed. "Just right," she whispered and fell fast asleep.

Meanwhile, the owners of the cottage, three bears—Papa Bear, Mama Bear, and Baby Bear—returned home from a walk. They immediately noticed something was off. "Someone’s been eating my porridge!" growled Papa Bear. "Someone’s been eating my porridge too!" said Mama Bear. "Someone’s been eating my porridge and ate it all up!" cried Baby Bear. Moving to the living room, they found the chairs. "Someone’s been sitting in my chair!" rumbled Papa Bear. "Someone’s been sitting in my chair too!" said Mama Bear. "Someone’s been sitting in my chair and broke it!" wailed Baby Bear.

The bears climbed the stairs and found their beds disturbed. "Someone’s been sleeping in my bed!" boomed Papa Bear. "Someone’s been sleeping in my bed too!" said Mama Bear. "Someone’s been sleeping in my bed, and she’s still here!" exclaimed Baby Bear. Goldilocks woke up to see three bears staring at her. Startled and frightened, she jumped out of the bed, ran down the stairs, and dashed out of the house as fast as her legs could carry her. She never returned to the forest again, and the bears lived peacefully in their cottage once more. Goldilocks, on the other hand, learned an important lesson: never enter a stranger’s house uninvited!`;

function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'train'>('generate');
  const [text, setText] = useState('');
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>('');
  const [voiceName, setVoiceName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastGeneration, setLastGeneration] = useState<{ id: string; numChunks: number } | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isHighQuality, setIsHighQuality] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const savedGeneration = localStorage.getItem('lastGeneration');
    const savedText = localStorage.getItem('lastText');
    if (savedGeneration) {
      setLastGeneration(JSON.parse(savedGeneration));
    }
    if (savedText) {
      setText(savedText);
    }
  }, []);

  const fetchVoices = async () => {
    try {
      setIsRefreshing(true);
      const response = await getVoices();
      setVoices(response.voices);
      if (!selectedVoice && response.voices.length > 0) {
        setSelectedVoice(response.voices[0]);
      }
    } catch (err) {
      setError('Failed to fetch voices');
      setTimeout(() => setError(null), 2000);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchVoices();
  }, []);

  const startRecording = async () => {
    try {
      chunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm',
      });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setRecordingTime(0);

      // Start the timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Could not access microphone');
      setTimeout(() => setError(null), 2000);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], `${voiceName}.webm`, { type: 'audio/webm' });
        handleCreateVoice(audioFile);
      };
      setIsRecording(false);

      // Clear the timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Stop all tracks
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleCreateVoice = async (audioFile: File) => {
    if (!voiceName) return;

    setIsLoading(true);
    setError(null);

    try {
      await createVoiceProfile(voiceName, audioFile);
      setVoiceName('');
      alert('Voice profile created successfully!');
      fetchVoices(); // Refresh the voice list after creating a new voice
    } catch (err) {
      setError('Failed to create voice profile');
      setTimeout(() => setError(null), 2000);
    } finally {
      setIsLoading(false);
    }
  };

  const playAudioSequentially = async (generationId: string, numChunks: number) => {
    setCurrentlyPlaying(true);

    for (let i = 1; i <= numChunks; i++) {
      const partNumber = i.toString().padStart(3, '0');
      const audioUrl = await getAudioUrl(generationId, `part${partNumber}.wav`);

      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        await audioRef.current.play();

        await new Promise((resolve) => {
          if (audioRef.current) {
            audioRef.current.onended = resolve;
          }
        });
      }
    }

    setCurrentlyPlaying(false);
  };

  const handleGenerateSpeech = async () => {
    if (!text || !selectedVoice) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await generateSpeech(text, selectedVoice, isHighQuality);
      const generationInfo = {
        id: response.generation_id,
        numChunks: response.num_chunks
      };
      setLastGeneration(generationInfo);
      localStorage.setItem('lastGeneration', JSON.stringify(generationInfo));
      localStorage.setItem('lastText', text);

      await playAudioSequentially(response.generation_id, response.num_chunks);
    } catch (err) {
      setError('Failed to generate speech');
      setTimeout(() => setError(null), 2000);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReplay = async () => {
    if (!lastGeneration) return;

    setCurrentlyPlaying(true);
    try {
      await playAudioSequentially(lastGeneration.id, lastGeneration.numChunks);
    } catch (err) {
      setError('Failed to replay speech');
      setTimeout(() => setError(null), 2000);
    }
    setCurrentlyPlaying(false);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="w-full max-w-2xl space-y-8">
          <div className="text-center">
            <img src="/logo.svg" alt="Vocazee Logo" className="mx-auto mb-4 w-72" />
            <p className="text-gray-500">Generate natural-sounding speech from text</p>
          </div>

          <div className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-100">
            <div className="flex border-b border-gray-100">
              <button
                className={`tab ${activeTab === 'generate' ? 'active' : ''}`}
                onClick={() => setActiveTab('generate')}
              >
                <span className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4" />
                  Generate Speech
                </span>
              </button>
              <button
                className={`tab ${activeTab === 'train' ? 'active' : ''}`}
                onClick={() => setActiveTab('train')}
              >
                <span className="flex items-center gap-2">
                  <Mic className="w-4 h-4" />
                  Train Custom Voice
                </span>
              </button>
            </div>

            <div className="p-6 space-y-6">
              {activeTab === 'generate' ? (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="block text-sm font-medium text-gray-700">Voice</label>
                      <div className="flex items-center gap-2">
                        <select
                          value={selectedVoice}
                          onChange={(e) => setSelectedVoice(e.target.value)}
                          className="input-field"
                        >
                          <option value="">Select a voice</option>
                          {voices.map((voice) => (
                            <option key={voice} value={voice}>
                              {voice}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={fetchVoices}
                          className="p-2 text-gray-500 hover:text-[#6320EE] transition-colors"
                          disabled={isRefreshing}
                        >
                          <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="block text-sm font-medium text-gray-700">Text</label>
                      <div 
                        className="relative inline-flex h-7 bg-gray-200 rounded-full cursor-pointer w-32 items-center"
                        onClick={() => setIsHighQuality(!isHighQuality)}
                      >
                        <div className="absolute inset-0 flex items-center justify-between px-2 text-xs font-medium">
                          <span className={`z-10 ${!isHighQuality ? 'text-white' : 'text-gray-500'}`}>Speed</span>
                          <span className={`z-10 ${isHighQuality ? 'text-white' : 'text-gray-500'}`}>Quality</span>
                        </div>
                        <div 
                          className={`absolute h-7 w-16 bg-[#6320EE] rounded-full transition-transform duration-200 transform ${
                            isHighQuality ? 'translate-x-16' : 'translate-x-0'
                          }`}
                        />
                      </div>
                    </div>
                    <textarea
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      className="input-field min-h-[120px]"
                      placeholder="Enter text to convert to speech..."
                    />
                  </div>

                  <div className="flex gap-2">
                    <button
                      className="btn-primary w-full flex items-center justify-center gap-2"
                      onClick={handleGenerateSpeech}
                      disabled={!text || !selectedVoice || isLoading || currentlyPlaying}
                    >
                      <Play className="w-4 h-4" />
                      {currentlyPlaying ? 'Playing...' : 'Generate Speech'}
                    </button>
                    {lastGeneration && (
                      <button
                        onClick={handleReplay}
                        disabled={currentlyPlaying}
                        className="btn-primary flex items-center justify-center gap-2 !bg-[#4C1BA6]"
                      >
                        <Volume2 className="w-4 h-4" />
                        Replay
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-700">Voice Name</label>
                    <input
                      type="text"
                      value={voiceName}
                      onChange={(e) => setVoiceName(e.target.value)}
                      className="input-field"
                      placeholder="Enter a name for your voice..."
                    />
                  </div>

                  <div className="bg-gray-50 rounded-lg p-6 space-y-4">
                    <p className="mb-4 text-[#6320EE] font-medium">
                      Read this text aloud. Keep your tone and volume natural. Please ensure you have a good microphone and are in a quiet environment for the best quality recording. For best results, read for 3 minutes.
                    </p>
                    <div className="h-[400px] overflow-y-auto pr-4 space-y-6 text-gray-700 font-medium leading-relaxed">
                      <div>
                        <h2 className="text-2xl font-bold text-[#6320EE] mt-4 mb-2">The Three Little Pigs</h2>
                        {TRAINING_TEXT_1}
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-[#6320EE] mt-8 mb-2">Goldilocks and the Three Bears</h2>
                        {TRAINING_TEXT_2}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <button
                      onClick={isRecording ? stopRecording : startRecording}
                      disabled={!voiceName}
                      className={`btn-primary flex items-center gap-2 ${isRecording ? 'bg-red-500 hover:bg-red-600' : ''}`}
                    >
                      {isRecording ? (
                        <>
                          <StopCircle className="w-4 h-4" />
                          Stop Recording ({Math.floor(recordingTime / 60)}:{String(recordingTime % 60).padStart(2, '0')})
                        </>
                      ) : (
                        <>
                          <Mic className="w-4 h-4" />
                          Start Recording
                        </>
                      )}
                    </button>
                    {isLoading && <p className="text-gray-500">Processing voice...</p>}
                  </div>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-lg">
              {error}
            </div>
          )}

          <audio ref={audioRef} className="hidden" />
        </div>
      </main>

      <footer className="py-4 text-center text-sm text-gray-500 border-t border-gray-100">
        Copyright 2025 Vocazee. All Rights Reserved | Built by Taeef Najib with <span className="text-red-500">♥</span>
      </footer>
    </div>
  );
}

export default App;