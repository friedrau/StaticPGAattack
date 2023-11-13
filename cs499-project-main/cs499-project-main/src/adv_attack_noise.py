import argparse
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import wandb
from adv_attack import ASRAdversarialExampleGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
  # set seed
  torch.manual_seed(42)

  parser = argparse.ArgumentParser(description='Adversarial Attack')
  parser.add_argument('--config', type=str, default='config/config_noise.yaml', help='Path to the config file.')
  parser.add_argument('--output', type=str, default='data/output_noise.wav', help='Path to the output audio file.')
  parser.add_argument('--length', type=int, default=10, help='Audio length in seconds.')
  args = parser.parse_args()

  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  asr_model = config['model']
  target_transcript = config['target']

  alpha = config['alpha']
  epsilon = config['epsilon']
  iterations = config['iterations']

  # Generate 10 seconds of white noise
  noise_range = 0.1
  noise = torch.randn(1, 16000 * args.length).unsqueeze(0) * noise_range
  audio_input = noise.to(device)

  torchaudio.save(args.output.replace('.wav', '_input.wav'), audio_input[0].cpu(), 16000)

  generator = ASRAdversarialExampleGenerator(asr_model, target_transcript, alpha, epsilon, iterations)

  wandb.init(
    project='cs499-project',
    config={
      'target_transcript': target_transcript,
      'alpha': alpha,
      'epsilon': epsilon,
    },
    mode='online' if config['wandb']['enabled'] else 'disabled',
  )

  adversarial_waveform, delta = generator.generate_adversarial_example(audio_input)
  torchaudio.save(args.output.replace('.wav', '_output.wav'), adversarial_waveform, 16000)
  torchaudio.save(args.output.replace('.wav', '_delta.wav'), delta, 16000)

  wandb.finish()
