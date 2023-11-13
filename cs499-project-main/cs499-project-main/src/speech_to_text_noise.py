import argparse

import torch
import torch.nn as nn
import torchaudio
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

from utils import calculate_cer

#sns.set()


def load_pretrained_model(model_name):
  processor = Wav2Vec2Processor.from_pretrained(model_name)
  model = Wav2Vec2ForCTC.from_pretrained(model_name)
  return processor, model


def load_audio(audio_file):
  audio_input, input_sample_rate = torchaudio.load(audio_file)

  # Convert stereo to mono if necessary
  if audio_input.ndim > 1:
    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

  # Resample the audio to 16000 Hz if necessary
  if input_sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(input_sample_rate, 16000)
    audio_input = resampler(audio_input)

  return audio_input


def fftnoise(f):
  f = np.array(f, dtype='complex')
  Np = (len(f) - 1) // 2
  phases = np.random.rand(Np) * 2 * np.pi
  phases = np.cos(phases) + 1j * np.sin(phases)
  f[1:Np+1] *= phases
  f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
  return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
  freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
  f = np.zeros(samples)
  idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
  f[idx] = 1
  return torch.from_numpy(fftnoise(f)).float().unsqueeze(0)


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
  freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
  f = np.zeros(samples)
  idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
  f[idx] = 1
  return torch.from_numpy(fftnoise(f)).float().unsqueeze(0)


def generate_noise_between_two_herts(audio_input, epsilon, low_freq, high_freq):
    #return torch.randn_like(audio_input) * epsilon
    return band_limited_noise(low_freq, high_freq, samples=audio_input.shape[1], samplerate=16000) * 320 * epsilon


def generate_noise_limit_high_bandwidth(audio_input, epsilon, low_freq, high_freq):
   low_freq_noise = band_limited_noise(0, low_freq, samples=audio_input.shape[1], samplerate=16000) * 320 * epsilon
   high_freq_noise = band_limited_noise(low_freq, high_freq, samples=audio_input.shape[1], samplerate=16000) * 320 * epsilon / 30
   return low_freq_noise + high_freq_noise
def generate_noise_limit_low_bandwidth(audio_input, epsilon, low_freq, high_freq):
   low_freq_noise = band_limited_noise(0, low_freq, samples=audio_input.shape[1], samplerate=16000) * 320 * epsilon / 30
   high_freq_noise = band_limited_noise(low_freq, high_freq, samples=audio_input.shape[1], samplerate=16000) * 320 * epsilon 
   return low_freq_noise + high_freq_noise


def generate_noise_real(audio_input, epsilon, low_freq, high_freq):
  #load real noise from file
  noise, sample_rate = torchaudio.load('real_noise.wav')
  return noise * epsilon


def create_spectrogram(wavfile, outputfile=None):
  # Load audio
  waveform, sample_rate = torchaudio.load(wavfile)

  # Create a spectrogram from the waveform
  spectrogram = torchaudio.transforms.Spectrogram()(waveform)

  # Convert the spectrogram to numpy array for plotting
  spectrogram_np = spectrogram.numpy()

  # Get number of frames from the spectrogram shape
  num_frames = spectrogram_np.shape[2]

  # Get the time axis values
  duration = waveform.shape[1] / sample_rate
  time_axis = np.linspace(0, duration, num_frames)

  # Plot and save the spectrogram
  plt.figure(figsize=(10, 4))
  plt.imshow(np.log(spectrogram_np[0, :, :]), aspect='auto', origin='lower',
             vmin=-10, vmax=5, extent=[time_axis.min(), time_axis.max(), 0, sample_rate/2])
  plt.title('Spectrogram of ' + wavfile)
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency Bin')
  plt.colorbar(format="%+2.0f dB")

  if outputfile is not None:
    plt.savefig(outputfile)

  #plt.show()


def loss_CTC(processor, model, x, y):
  logits = model(x).logits
  logits = torch.nn.functional.pad(logits, (0, y.shape[1] - logits.shape[2]))
  logits = logits.to(torch.float32)
  y = y.to(torch.float32)

  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)

  preds = logits.log_softmax(-1)
  batch, seq_len, classes = preds.shape
  preds = preds.permute(1, 0, 2)
  target_lengths = torch.count_nonzero(y, axis=1)

  pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)

  ctc_loss = nn.CTCLoss(zero_infinity=True, reduction='none')
  loss = ctc_loss(preds, y, pred_lengths, target_lengths)

  return loss, transcription


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=str, help='Path to the audio file to transcribe')
  parser.add_argument('--config', type=str, default='config/config_noise.yaml', help='Path to the config file used for generating this audio file.')
  parser.add_argument('--model', type=str, default='facebook/wav2vec2-base-960h', help='Name of the pretrained model to use')
  args = parser.parse_args()

  # set seeds
  torch.manual_seed(42)
  np.random.seed(42)

  model_name = args.model
  audio_file = args.input

# noise_range = np.arange(0, 0.1, 0.005)
  noise_range = np.arange(0, .5, 0.01)
  Freq_range = np.arange(0, 8000, 1000)
  Freq_range2 = np.arange(1000, 9000, 1000)

  processor, model = load_pretrained_model(model_name)
  audio_input = load_audio(audio_file)

  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  adversarial_transcript = config['target']
  adversarial_input_ids = processor(text=adversarial_transcript, return_tensors='pt').input_ids

  #losses = []
  #cers = []

  low_freq = 0000
  high_freq = 8000

  for i, (freq) in enumerate(Freq_range):
    losses = []
    cers = []
    for noise in noise_range:

      noise_clip = generate_noise_between_two_herts(audio_input, noise, Freq_range[i], Freq_range2[i])
      adv_loss, transcription = loss_CTC(processor, model, audio_input + noise_clip, adversarial_input_ids)
      
      transcription = transcription[0]

      cer = calculate_cer(adversarial_transcript, transcription)

      losses.append(adv_loss.item())
      cers.append(cer)

      print(f'**** Noise = {noise} ****')
      print(f'Transcription: {transcription}')
      print(f'Adv Loss: {adv_loss.item()}; CER: {cer}')
      print()
  
    #saveNameSpectrogram = 'spectrogram_freq_' + str(freq) + 'noise_'+str(noise)+'_.png'
    #torchaudio.save(f'temp_noise_freq_' + str(freq) + 'noise_' + str(noise) + "_.wav", noise_clip, 16000)
    #create_spectrogram('temp_noise_freq_' + str(freq) + 'noise_' + str(noise) + "_.wav",saveNameSpectrogram)

    plt.figure()
    plt.plot(noise_range, cers, 'o')
    plt.xlabel('Noise')
    plt.ylabel('CER')
    plt.title(f'CER vs. Noise limited to '+str(Freq_range[i])+'Hz'+' - '+str(Freq_range2[i])+'Hz')

    saveName = 'CER_simulated_freq_' + str(freq) + '_.png'
    plt.savefig(saveName)

    plt.figure()
    plt.plot(noise_range, losses, 'o')
    plt.xlabel('Noise')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Noise limited to '+str(Freq_range[i])+'Hz'+' - '+str(Freq_range2[i])+'Hz')

    saveName = 'Loss_simulated_freq_' + str(freq) + '_.png'
    plt.savefig(saveName)

  # Save last noise clip
  torchaudio.save(f'noise_real.wav', noise_clip, 16000)
  torchaudio.save(f'input_noise_real.wav', audio_input + noise_clip, 16000)

  plt.show()
