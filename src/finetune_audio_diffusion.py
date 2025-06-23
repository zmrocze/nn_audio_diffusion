
from diffusers import DiffusionPipeline
import torch
# from prefigure.prefigure import push_wandb_config
import torch
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
import torchtune.training
import wandb
import librosa
import torchtune
from datasets import load_from_disk
from peft import LoraConfig, inject_adapter_in_model
from dataclasses import dataclass
# import torch_audiomentations as taug
import torchaudio as ta
import audiomentations as aug
import audiomentations as taug
import random
# import diffusers
import math
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import RichProgressBar


models_map = {
  "glitch-440k": {
    'sample_rate': 48000,
    'sample_size': 65536
    },
  "jmann-small-190k": {
    'sample_rate': 48000,
    'sample_size': 65536
    },
  "jmann-large-580k": {
    'sample_rate': 48000,
    'sample_size': 131072
    },
  "maestro-150k": {
    'sample_rate': 16000,
    'sample_size': 65536
    },
  "unlocked-250k": {
  'sample_rate': 16000,
  'sample_size': 65536
  },
  "honk-140k": {
    'sample_rate': 16000,
    'sample_size': 65536
    },
}

@dataclass
class TrainingConfig:
    model_name = "unlocked-250k"
    data_path = "./filter_fma_rock"
    sample_size = models_map[model_name]['sample_size']
    sample_rate = models_map[model_name]['sample_rate']
    data_loader_num_workers = 4
    batch_size = 32
    # eval_batch_size = 16  # how many images to sample during evaluation
    gradient_accumulation_steps = 1
    lora_rank = 16  # the rank of the LoRA layers1
    lora_alpha = 16 # scaling factor, which seems hardly necessary
    lr = 4e-5
    lr_warmup_steps = 5 # epochs
    num_cycles=0.5  # cosine annealing cycles
    num_epochs = 1000
    # save_image_epochs = 10
    save_model_epochs = 30
    demo_every = 5
    n_samples = 5
    save_demo = True
    demo_save_path = './demo_songs'
    ckpt_load_path = None # 'best', 'last', <path]>
    wandb_log_model = 'all'
    check_val_every_n_epoch = 5
    # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = f"ddim-lora-{model_name}"  # the model name locally and on the HF Hub
    name = f"ddim-lora-{model_name}-{random.randint(0, 1000)}"  # the name of the wandb run
    project_name = "nn-audio-diffusion"
    save_path = f'{name}-ckpt'
    # overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    # seed = 42

config = TrainingConfig()

def optional(x, bool):
  if bool:
    return [x]
  else:
    return []

# samples_per_s = 44100
def audio_augmentations(sample_size, sample_rate, use_train_augs = False):
  output_type = 'tensor'
  augmentation = aug.Compose(
    transforms=
      [aug.AdjustDuration(duration_samples=sample_size, p=1.0)]
      +
      optional(
        aug.TimeStretch(
            min_rate=0.8,
            max_rate=1.25,
            leave_length_unchanged=True,
            p=0.8
        ),
        use_train_augs)
      + optional(
        aug.OneOf(
          transforms=[
              taug.LowPassFilter(
                  min_cutoff_freq=500.0,
                  max_cutoff_freq=2000.0,
                  p=1.0,
                  # output_type=output_type
              ),
              taug.HighPassFilter(
                  min_cutoff_freq=100.0,
                  max_cutoff_freq=2400.0,
                  p=1.0,
                  # output_type=output_type
              ),
              taug.BandPassFilter(
                  # min_center_frequency=200.0,
                  # max_center_frequency=4000.0,
                  min_center_freq=200.0,
                  max_center_freq=4000.0,
                  p=1.0,
                  # output_type=output_type
              ),
              aug.ClippingDistortion(min_percentile_threshold=0.01,
                  max_percentile_threshold=0.99,
                  p=0.5),
              # aug.Limiter(
              #   min_threshold_db=-16.0,
              #   max_threshold_db=0.0,
              #   threshold_mode="relative_to_signal_peak",
              #   p=1.0,
              # ), # todo: cylimiter dependency problem
              taug.PitchShift(
                # min_transpose_semitones=-5.0,
                # max_transpose_semitones=5.0,
                min_semitones=-5.0,
                max_semitones=5.0,
                # sample_rate=sample_rate,
                p=1.0,
                # output_type=output_type
              ),
          ],
          p=1.0
        ),
        use_train_augs)
      +
      optional(
        aug.SomeOf( transforms=[
          aug.TanhDistortion(
            min_distortion=0.01,
            max_distortion=0.15,
            p=0.8
          ),
          aug.RepeatPart(mode="insert", p=0.8),
        ],
        num_transforms=(0,None),
        ),
        use_train_augs)
      +
      optional(
        aug.Gain(
          min_gain_db=-15.0,
          max_gain_db=5.0,
          # p=0.5,
          p=0.5,
        )
        , use_train_augs)
      +
      # optional(
      #   taug.ShuffleChannels(p=0.5, 
      #     # output_type=output_type
      #     ),
      #   use_train_augs)
      # +
      
      # taug.OneOf(transforms=[
      #   taug.PeakNormalization(apply_to="only_too_loud_sounds", p=1.0, output_type=output_type),
        # [taug.PeakNormalization(apply_to="all", p=1.0
        #                         # , output_type=output_type
        #                         )]
        [aug.Normalize(apply_to="only_too_loud_sounds", p=1.0)]
      # ], output_type=output_type),
      +
      optional(
        aug.PolarityInversion(p=0.5),
      use_train_augs)
  )

  # def transforms(examples):
  #   # print(len(examples['audio']), "audio samples")
  #   x = [augmentation(aud['array'], sample_rate=aud['sampling_rate']) for aud in examples['audio']]
  #   examples["audio"] = x
  #   print("Transformed audio samples:", [aud.shape for aud in examples['audio']])
  #   return examples
  
  # return transforms, augmentation
  return augmentation

class MusicDataset(Dataset):
  def __init__(self, hf_dataset, transform=None):
    self.dataset = hf_dataset
    self.transform = transform
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    item = self.dataset[idx]
    audio = item['audio']['array']
    if self.transform:
      audio = self.transform(audio, sample_rate=item['audio']['sampling_rate'])
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    return audio_tensor

def make_dataloaders(data, config=config):
  sample_size=config.sample_size
  sample_rate=config.sample_rate
  num_workers=config.data_loader_num_workers
  batch_size=config.batch_size
  test_transformation = audio_augmentations(sample_size, sample_rate, use_train_augs = False)
  train_dataset = MusicDataset(data['train'], transform=audio_augmentations(sample_size, sample_rate, use_train_augs = True))
  val_dataset = MusicDataset(data['validation'], transform=test_transformation)
  test_dataset = MusicDataset(data['test'], transform=test_transformation)

  dl = lambda ds: torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  return {
    'train': dl(train_dataset),
    'validation': dl(val_dataset),
    'test': dl(test_dataset),
    'genres': data['genres'],
  }

def count_n_params(model):
  """Counts the number of trainable parameters in a model."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_lora(model, config=config):
  """ mutating function """
  num_learnable_params = count_n_params(model)
  num_learnable_modules = sum(1 for p in model.parameters() if p.requires_grad)
  unet_lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    init_lora_weights=True, # "gaussian",
    # modules_to_save=["lm_head"],
    # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    target_modules="all-linear",
    bias="lora_only",
  )

  _model = inject_adapter_in_model(unet_lora_config, model)

  print(f"Before LoRA: {num_learnable_params} learnable params across {num_learnable_modules} modules.")
  print(f"After LoRA: {count_n_params(model)} learnable params across {sum(1 for p in model.parameters() if p.requires_grad)} modules.")

  return _model


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


class DiffusionUncond(pl.LightningModule):
    def __init__(self, pipe, config): # todo: config
      super().__init__()
      self.pipe = pipe
      self.config = config
      self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=42)
        
    def configure_optimizers(self):
      optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        #  torchtune.modules.get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = - 1) → LambdaLR[source]
      lr_schedule = torchtune.training.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=config.num_epochs, num_cycles=0.5, last_epoch=-1)
      return {
        "optimizer": optimizer, 
        "lr_scheduler": {
            "scheduler": lr_schedule,
            "interval": "epoch",  # or "epoch"
            "frequency": 1,
          }
        }
  
    @property
    def model(self):
      return self.pipe.unet    
    @property
    def scheduler(self):
      return self.scheduler # not self.pipe.scheduler
    def sample_rate(self):
      return self.pipe.unet.sample_rate
    
    def log_some_samples(self, step, n_samples=10, save_demo=True):
      "Log images to wandb and save them to disk"
      audios = self.pipe(audio_length_in_s=4, batch_size=n_samples)
      # spectrogram
      log_dict = {}
      filename = f'{config.demo_save_path}/demo_{step}.wav'
      if save_demo:
        torchaudio.save(filename, audios, self.sample_rate)

      log_dict["sampled_audio"] = {
        f"epoch_{step}": 
          [ 
            { 
              "audio":
                wandb.Audio(au, sample_rate=self.sample_rate, caption=f"filename_{i}"),
              "mel":
                wandb.Image(librosa.feature.melspectrogram(au), caption=f"filename_{i}")
            # wandb.Audio(audios, sample_rate=self.sample_rate, caption=f"filename")
            } 
            for i, au in enumerate(audios)
          ]
      }

      self.log_dict(log_dict, step=step)

    # def on_validation_epoch_end(self):
    #   # Log some images to wandb
    #   self.log_some_samples(n_samples=config.eval_batch_size, save_demo=True)

    def forward_dance_diffusion(self, batch):
        reals = batch.unsqueeze(1)
        reals = torch.concat([reals, reals.detach()], dim=1) # make stereo

        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        t = get_crash_schedule(t)
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)
        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        v = self.model(noised_reals, t).sample
        loss = F.mse_loss(v, targets)
        return loss

    def diffusion_bad(self, batch):
      audio = batch # (batch_size, sample_size)
      batch_shape = audio.shape
      batch_size = audio.shape[0]

      noise = torch.randn(batch_shape, device=self.device)
      timesteps = torch.randint(
          0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device,
          dtype=torch.int64
      )
      # Add noise to the clean images according to the noise magnitude at each timestep
      # (this is the forward diffusion process)
      noisy_audio = self.scheduler.add_noise(audio, noise, timesteps)

      noise_pred = self.model(noisy_audio, timesteps).sample

      # self.scheduler.step(noise_pred, )

      return noise, noise_pred

    def training_step(self, batch, batch_idx):
        
        # noise, noise_pred = self.random_timestep_forward(batch)
        # loss = F.mse_loss(noise_pred, noise)
        loss = self.forward_dance_diffusion(batch)
        self.log_dict({
          'train/loss': loss.detach()
          }, prog_bar=True
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        # noise, noise_pred = self.random_timestep_forward(batch)
        # loss = F.mse_loss(noise_pred, noise)
        loss = self.forward_dance_diffusion(batch)
        self.log_dict({
          'val/loss': loss.detach()
          }, prog_bar=True
        )

        return loss
    
    def metrics(self):
      pass

class DemoCallback(pl.Callback):
  def __init__(self, config):
      super().__init__()
      self.demo_every = config.demo_every
      self.n_samples = config.n_samples
      self.save_demo = config.save_demo
  
  def on_train_start(self, trainer, pl_module):
    pl_module.log_some_samples(trainer.global_step, self.n_samples, save_demo=True)

  def on_train_epoch_end(self, trainer, pl_module):
    if (trainer.current_epoch + 1) % self.demo_every == 0:
      pl_module.log_some_samples(trainer.global_step, self.n_samples, save_demo=True)


def main(config=config):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  data = load_from_disk(config.data_path)
  data_loaders = make_dataloaders(data, config)
  pipe = DiffusionPipeline.from_pretrained(f"harmonai/{config.model_name}")
  pipe.to(device)
  wandb_logger = pl.loggers.WandbLogger(project=config.project_name, name=config.name, log_model=config.wandb_log_model)
  save_on_exc = pl.callbacks.OnExceptionCheckpoint(f"{config.save_path}/exc_save",)
  ckpt_callback = pl.callbacks.ModelCheckpoint(
    every_n_epochs=1,
    save_on_train_epoch_end=False, # together with every_n_epochs and check_val_every_n_epoch, this will save the model on validation, that is on check_val_every_n_epoch
    auto_insert_metric_name=True,
    mode='min',
    monitor='val/loss',
    save_top_k=2, dirpath=config.save_path
  )
  demo_callback = DemoCallback(config)

  apply_lora(pipe.unet, config)
  model = DiffusionUncond(pipe, config)

  wandb_logger.watch(model, log="all")
  wandb_logger.experiment.config.update(config)

  trainer = pl.Trainer(
      # precision=16,
      accumulate_grad_batches=config.gradient_accumulation_steps,
      callbacks=[ckpt_callback, demo_callback, RichProgressBar(), save_on_exc],
      logger=wandb_logger,
      check_val_every_n_epoch=config.check_val_every_n_epoch,
      # log_every_n_steps=1,
      max_epochs=config.num_epochs,
    )

  trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'], ckpt_path=config.ckpt_load_path)

main()
