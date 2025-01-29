export interface CreateVoiceResponse {
  status: string;
  voice_name: string;
  num_chunks: number;
}

export interface GenerateSpeechResponse {
  status: string;
  generation_id: string;
  num_chunks: number;
}

export interface ErrorResponse {
  detail: string;
}