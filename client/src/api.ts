const BASE_URL = 'http://127.0.0.1:8000';

export async function getVoices() {
  const response = await fetch(`${BASE_URL}/voices`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch voices');
  }

  return response.json();
}

export async function createVoiceProfile(voiceName: string, file: File) {
  const formData = new FormData();
  formData.append('voice_name', voiceName);
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/create-voice`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to create voice profile');
  }

  return response.json();
}

export const generateSpeech = async (text: string, voiceName: string, isHighQuality: boolean = false) => {
  const response = await fetch(`${BASE_URL}/generate-speech`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      voice_name: voiceName,
      high_quality: isHighQuality
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to generate speech');
  }

  return response.json();
};

export async function getAudioUrl(generationId: string, part: string): Promise<string> {
  return `${BASE_URL}/audio/${generationId}/${part}`;
}