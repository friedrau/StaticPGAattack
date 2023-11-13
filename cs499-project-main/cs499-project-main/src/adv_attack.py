import argparse
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import wandb
from utils import calculate_cer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_audio_file(audio_file):
  audio_input, input_sample_rate = torchaudio.load(audio_file)

  # Convert stereo to mono if necessary
  if audio_input.ndim > 1:
    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

  # Resample the audio to 16000 Hz if necessary
  if input_sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(input_sample_rate, 16000)
    audio_input = resampler(audio_input)

  audio_input = audio_input.unsqueeze(0)

  return audio_input


class ASRAdversarialExampleGenerator(pl.LightningModule):

  def __init__(self, asr_model_name, target_transcript, alpha, epsilon, iterations):
    super().__init__()
    self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name).to(device)
    self.processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
    self.target_transcript = target_transcript
    self.target_transcript_text =target_transcript

    self.alpha = alpha
    self.epsilon = epsilon
    self.iterations = iterations


  def forward(self, x):
    return self.asr_model(x)


  def pgd_attack(self, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True).to(device)

    for t in range(num_iter):
      loss, transcription = self.loss_CTC(x + delta, y)

      # backward pass for computing the gradients of the loss w.r.t to learnable parameters
      loss.backward()

      delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
      delta.grad.zero_()

      cer = calculate_cer(self.target_transcript_text, transcription[0])

      wandb.log({ 'loss': loss.item(), 'cer': cer, 'transcription': transcription[0], 'iteration': t + 1 })

      # print loss and transcription every 10 iterations
      if t % 10 == 0:
        logger.info('****--------------------------****')
        logger.info(f'Loss at iteration {t}: {loss.item()}')
        logger.info(f'CER at iteration {t}: {cer}')
        logger.info('Transciption: ' + str(transcription[0]))
        logger.info('****--------------------------****')

    return delta.detach()


  def loss_CTC(self, x, y):
    logits = self.asr_model(x.squeeze(0)).logits
    logits = torch.nn.functional.pad(logits, (0, y.shape[1] - logits.shape[2]))
    logits = logits.to(torch.float32)
    y = y.to(torch.float32)

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = self.processor.batch_decode(predicted_ids)

    preds = logits.log_softmax(-1)
    batch, seq_len, classes = preds.shape
    preds = preds.permute(1, 0, 2)
    target_lengths = torch.count_nonzero(y, axis=1)

    pred_lengths = torch.full(size=(batch,), fill_value=seq_len, dtype=torch.long)

    ctc_loss = nn.CTCLoss(zero_infinity=True, reduction='none')
    loss = ctc_loss(preds, y, pred_lengths, target_lengths)

    return loss, transcription


  def generate_adversarial_example(self, audio_input):
    target_transcript = self.processor(text=self.target_transcript, return_tensors='pt').input_ids.to(device)
    delta = self.pgd_attack(audio_input, target_transcript, self.epsilon, self.alpha, self.iterations)
    adversarial_input = audio_input + delta

    adversarial_transcript = self.processor.batch_decode(torch.argmax(self.asr_model(adversarial_input.squeeze(0)).logits, dim=-1))
    logger.info(f'Final transcription: {adversarial_transcript}')

    return adversarial_input[0].cpu(), delta[0].cpu()


if __name__ == '__main__':
  # set seed
  torch.manual_seed(42)
  np.random.seed(42)

  parser = argparse.ArgumentParser(description='Adversarial Attack')
  parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file.')
  parser.add_argument('--input', type=str, default='data/OSR_us_000_0035_8k.wav', help='Path to the input audio file.')
  parser.add_argument('--output', type=str, default='data/OSR_us_000_0035_8k_adv.wav', help='Path to the output audio file.')
  parser.add_argument('--noise', type=float, default=None, help='Add noise to the input audio file.')
  args = parser.parse_args()

  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  asr_model = config['model']
  target_transcript = config['target']

  alpha = config['alpha']
  epsilon = config['epsilon']
  iterations = config['iterations']

  audio_input = load_audio_file(args.input).to(device)

  if args.noise is not None:
    # add noise to audio input
    noise = torch.randn(audio_input.shape).to(device) * args.noise #0.04
    audio_input = audio_input + noise

  torchaudio.save('input.wav', audio_input[0].cpu(), 16000)

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

  adversarial_waveform, noise = generator.generate_adversarial_example(audio_input)
  torchaudio.save(args.output, adversarial_waveform, 16000)
  torchaudio.save(args.output.replace('.wav', '_noise.wav'), noise, 16000)

  wandb.finish()
