import argparse

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_pretrained_model(model_name):
  processor = Wav2Vec2Processor.from_pretrained(model_name)
  model = Wav2Vec2ForCTC.from_pretrained(model_name)
  return processor, model


def transcribe_audio(processor, model, audio_file):
  audio_input, input_sample_rate = torchaudio.load(audio_file)

  # Convert stereo to mono if necessary
  if audio_input.ndim > 1:
    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

  # Resample the audio to 16000 Hz if necessary
  if input_sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(input_sample_rate, 16000)
    audio_input = resampler(audio_input)

  logits = model(audio_input).logits
  predicted_ids = torch.argmax(logits, dim=-1)

  transcription = processor.batch_decode(predicted_ids)
  return transcription[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, help='Path to the audio file to transcribe')
  parser.add_argument('--model', type=str, default='facebook/wav2vec2-base-960h', help='Name of the pretrained model to use')
  args = parser.parse_args()

  model_name = args.model
  audio_file = args.input

  processor, model = load_pretrained_model(model_name)
  transcription = transcribe_audio(processor, model, audio_file)

  print(f'Transcription: {transcription}')
