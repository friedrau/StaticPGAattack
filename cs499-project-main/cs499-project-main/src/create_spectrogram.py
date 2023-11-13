import argparse

import matplotlib.pyplot as plt
import numpy as np
import torchaudio


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

  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Create a Spectrogram from a wav file.')
  parser.add_argument('wavfile', type=str, help='Input wav file path')
  parser.add_argument('--output', type=str, help='Output png file path')

  args = parser.parse_args()

  create_spectrogram(args.wavfile, args.output)
