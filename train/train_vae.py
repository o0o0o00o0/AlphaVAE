import os, shutil
import math
import logging
import random
from pathlib import Path
from typing import *

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# from ldm.modules.losses import LPIPSWithDiscriminator
# from losses import LPIPSWithDiscriminator
# from taming.modules.losses.vqperceptual import LPIPS
from losses import *
from utils import *
import cv2

if is_wandb_available():
    import wandb

from transformers import HfArgumentParser
from dataclasses import dataclass, field
from trl import TrlParser



# Logger prints only on the main process
logger = get_logger(__name__, log_level="INFO")


@dataclass
class TrainingArguments:
    # Model loading related
    pretrained_path: str = field(
        default=None,
        metadata={"help": "Local pretrained model path."},
    )

    # Dataset related (using local training directory only)
    train_data_dir: str = field(
        default=None, metadata={"help": "Local directory containing training images."}
    )
    num_eval: int = field(
        default=4, metadata={"help": "Number of images to use from the end of the dataset as evaluation data."}
    )

    # Output and cache
    output_dir: str = field(
        default="vae-model-finetuned",
        metadata={"help": "Output directory where model predictions and checkpoints will be saved."},
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for downloading models and datasets."}
    )

    # Random seed and resolution
    seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed for reproducible training."}
    )
    resolution: int = field(
        default=512, metadata={"help": "Resolution for input images; images will be resized accordingly."}
    )

    # Training hyperparameters
    train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per device during training."}
    )
    num_train_epochs: int = field(
        default=100, metadata={"help": "Total number of training epochs."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of steps to accumulate gradients before updating model parameters."}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing to save memory."}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Initial learning rate."}
    )
    scale_lr: bool = field(
        default=False, metadata={"help": "Whether to scale the learning rate based on the number of devices and batch size."}
    )
    lr_scheduler: str = field(
        default="constant", metadata={"help": "Type of learning rate scheduler."}
    )
    lr_warmup_steps: int = field(
        default=500, metadata={"help": "Number of warmup steps for the learning rate scheduler."}
    )

    # Logging and checkpointing
    logging_dir: str = field(
        default="logs", metadata={"help": "TensorBoard logging directory."}
    )
    mixed_precision: str = field(
        default=None,
        metadata={"help": "Whether to use mixed precision training. Options: no, fp16, bf16."},
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Platform to report logs (e.g., tensorboard, wandb)."},
    )
    checkpointing_steps: int = field(
        default=1,
        metadata={"help": "Save checkpoint every specified number of steps."},
    )
    checkpoints_total_limit: int = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to keep."},
    )
    validation_steps: int = field(
        default=2000, metadata={"help": "Run validation every specified number of epochs."}
    )
    tracker_project_name: str = field(
        default="vae-fine-tune", metadata={"help": "Project name for logging."}
    )

    # Loss weights
    kl_scale: Optional[float] = field(
        default=None, metadata={"help": "Weight for the KL divergence term."}
    )
    ref_kl_scale: Optional[float] = field(
        default=None, metadata={"help": "Weight for the KL divergence term."}
    )
    lpips_scale: Optional[float] = field(
        default=None, metadata={"help": "Weight for the LPIPS loss."}
    )
    validation_on_first_step: bool = field(
        default=False, metadata={"help": "Whether to run validation when the first step finished."}
    )
    generator_loss_weight: float = field(
        default=1
    )
    discriminator_loss_weight: float = field(
        default=1
    )
    gan_start_step: Optional[int] = field(
        default=None # None for No use
    )
    naive_mse_loss: bool = field(
        default=False, metadata={"help": "Whether to use naive MSE loss."}
    )
    blend_prob: float = field(
        default=0, metadata={"help": "train detail blend prob"}
    )


# Custom dataset: traverse all images in the specified directory
class LocalImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, prob=0):
        self.root_dir = root_dir
        self.prob = prob
        self.transform = transform
        self.image_paths = []
        # Traverse the directory and collect all image files
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(dirpath, filename))
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def resize(self, image: Image.Image) -> Image.Image:
        w, h = image.width, image.height
        area = w * h
        max_area = 1024 * 1024

        if area > max_area:
            scale = (max_area / area) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
            w, h = new_w, new_h

        crop_w = w - (w % 16)
        crop_h = h - (h % 16)

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        return image.crop((left, top, right, bottom))

    def blend_rgba_to_bg(self, fg_rgba: Image.Image) -> Image.Image:
        fg = np.array(fg_rgba).astype(np.float32) / 255.0  # (H, W, 4)
        rgb = fg[..., :3]
        alpha = fg[..., 3:4]
        H, W = fg.shape[:2]
        
        color = np.random.randint(0, 3, size=(1, 1, 3), dtype=np.uint8)  # shape: (1, 1, 3)
        bg = np.ones((H, W, 3), dtype=np.float32) * (color.astype(np.float32) / 2.0)

        blended = rgb * alpha + bg * (1.0 - alpha)
        blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
        blended = np.concatenate([blended, np.ones((H, W, 1), dtype=np.uint8) * 255], axis=-1)
        return Image.fromarray(blended)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGBA")
        if random.random() < self.prob:
            image = self.blend_rgba_to_bg(image)
        # image = self.resize(image)
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "file_path": img_path}

    
@torch.no_grad()
def log_validation(test_dataloader, vae, accelerator, weight_dtype, global_step):
    if not accelerator.is_main_process:
        return

    logger.info("Running validation...")

    all_images = []
    for batch in test_dataloader:
        inputs = batch["pixel_values"].to(device=vae.device, dtype=weight_dtype)
        outputs = vae(inputs).sample

        combined = torch.cat([batch["pixel_values"].cpu(), outputs.cpu()], dim=-1)
        combined = (combined * 0.5 + 0.5).clamp(0., 1.)
        all_images.append(combined)

    all_images = torch.cat(all_images)
    grid = torchvision.utils.make_grid(all_images, nrow=all_images.shape[0], padding=0)

    if grid.shape[0] == 4:
        white_bg = grid[:3, ...] * grid[3:] + 1 * (1 - grid[3:])
        black_bg = grid[:3, ...] * grid[3:]
        grid = torch.cat((white_bg, black_bg), dim=-2)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_image("Validation: Original and Reconstruction", grid, global_step)
        elif tracker.name == "wandb":
            tracker.log({"Validation: Original and Reconstruction": wandb.Image(grid)}, step=global_step)
        else:
            logger.warning(f"Tracker {tracker.name} logging is not implemented.")

    torch.cuda.empty_cache()



def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values}


def print_training_setting(args):
    # green on and red off
    on_str = "\033[1;32mON\033[0m"
    off_str = "\033[1;31mOFF\033[0m"
    logger.info("Module Settings:")
    logger.info(f"    PatchGAN:  {on_str if (args.gan_start_step is not None) else off_str}  (start from {args.gan_start_step}-th step)")
    logger.info(f"    Norm KL Loss:  {on_str if (args.kl_scale is not None) else off_str}  (weight={args.kl_scale})")
    logger.info(f"    Ref KL Loss:  {on_str if (args.ref_kl_scale is not None) else off_str}  (weight={args.ref_kl_scale})")
    logger.info(f"    LPIPS Loss:  {on_str if (args.lpips_scale is not None) else off_str}  (weight={args.lpips_scale})")
    logger.info(f"    Naive MSE Loss:  {on_str if args.naive_mse_loss else off_str}")
    

def main():
    # Use TrlParser to parse arguments with config
    parser = TrlParser(TrainingArguments)
    args, = parser.parse_args_and_config()
    
    if 1 < 0: # for class annotation in code editor
        args = TrainingArguments()

    # Set up logging directory
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,
    )

    # Configure logging only on the main process
    if accelerator.is_main_process:
        logging.basicConfig(
            format="\033[1;36m%(asctime)s - %(levelname)s - %(name)s\033[0m | %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=True)
        
        print_training_setting(args)
        
        logger.info(args)
        
        copy_code_files(os.path.dirname(os.path.abspath(__file__)), os.path.join(logging_dir, "codes"))

    
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Process {accelerator.process_index}: Local Loading | Accelerate Done", flush=True)

    # Load VAE (generator)
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.pretrained_path).to(torch.float32)
    vae.requires_grad_(True)
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    
    if args.ref_kl_scale is not None:
        ref_vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.pretrained_path).to(torch.float32)
        ref_vae.requires_grad_(False)
        ref_vae.eval()
    

    # Define image preprocessing transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Use custom dataset to traverse all images under train_data_dir, and split dataset into training and evaluation using num_eval
    full_dataset = LocalImageDataset(args.train_data_dir, transform=train_transforms, prob=args.blend_prob)
    if len(full_dataset) < args.num_eval:
        raise ValueError("Not enough images in the dataset to satisfy num_eval requirement.")
    train_dataset = torch.utils.data.Subset(
        full_dataset, list(range(0, len(full_dataset) - args.num_eval))
    )
    test_dataset = torch.utils.data.Subset(
        full_dataset, list(range(len(full_dataset) - args.num_eval, len(full_dataset)))
    )
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
    )
    
    
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / (args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes))
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Initialize LPIPS network (discriminator)    
    # lpips_net = LPIPSWithDiscriminator(50000, kl_weight=args.kl_scale, disc_weight=args.lpips_scale, disc_in_channels=4)
    # loss_module = RGBAVAELoss(reduce_mean=True)
    loss_module = RGBAVAELoss(
        reduce_mean=False, 
        use_naive_mse=args.naive_mse_loss,
        use_patchgan=args.gan_start_step is not None,
        use_lpips=args.lpips_scale is not None,
    )
    
    # Create separate optimizers for generator (VAE) and discriminator (LPIPS)
    optimizer_gen = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    # optimizer_disc = torch.optim.AdamW(lpips_net.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    # print(optimizer_disc.param_groups[0].keys())

    lr_scheduler_generator = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=0.5 if args.lr_scheduler=="cosine" else 1
    )
    
    
    print(f"Process {accelerator.process_index}: Local Loading | All Done", flush=True)

    # Prepare modules for distributed training with accelerator
    vae, optimizer_gen, train_dataloader, lr_scheduler_generator = accelerator.prepare(
        vae, optimizer_gen, train_dataloader, lr_scheduler_generator
    )
    # if accelerator.is_main_process:
    #     print(type(loss_module), loss_module.__dict__.keys())
    #     print(loss_module.kl_loss)
    #     print(loss_module.reconstruction_loss)
    #     print(loss_module.perceptual_loss)        
    # exit(1)
    loss_module: RGBAVAELoss
    
    print(f"Process {accelerator.process_index}: Preparing Done", flush=True)
    
    assert num_update_steps_per_epoch == len(train_dataloader)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    if args.gan_start_step is not None:
        accelerator_discriminator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            step_scheduler_with_optimizer=False,
        )
        optimizer_disc = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
        lr_scheduler_discriminator = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps - args.gan_start_step,
            num_cycles=0.5 if args.lr_scheduler=="cosine" else 1
        )
        loss_module, optimizer_disc, lr_scheduler_discriminator = accelerator_discriminator.prepare(loss_module, optimizer_disc, lr_scheduler_discriminator)
    else:
        loss_module = loss_module.to(device=accelerator.device, dtype=weight_dtype)

    # loss_module: RGBAVAELoss = loss_module.to(device=accelerator.device, dtype=weight_dtype)
    if args.ref_kl_scale is not None:
        ref_vae: AutoencoderKL = ref_vae.to(device=accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = vars(args)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num eval examples = {len(test_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Num Steps = {args.max_train_steps}")
        logger.info(f"  Iterations/Epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        # train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            logs = {}
            with accelerator.accumulate(vae):
                with accelerator.autocast():
                    target = batch["pixel_values"].to(weight_dtype)
                    fg_alpha = (1 + target[:, 3:]) / 2
                    bg_alpha = (1 - target[:, 3:]) / 2
                    target_black_bg_blending = target * fg_alpha - bg_alpha
                    target_white_bg_blending = target * fg_alpha + bg_alpha
                    target_black_bg_blending[:, 3] = 1.
                    target_white_bg_blending[:, 3] = 1.
                    
                    composed_target = torch.cat((target, target_black_bg_blending, target_white_bg_blending), dim=0)
                    
                    unwrapped_vae: AutoencoderKL = accelerator.unwrap_model(vae)
                    posterior: DiagonalGaussianDistribution = unwrapped_vae.encode(composed_target).latent_dist
                    posterior, posterior_black, posterior_white = map(DiagonalGaussianDistribution, torch.chunk(posterior.parameters, 3, dim=0))
                    
                    if args.ref_kl_scale is not None:
                        ref_posterior: DiagonalGaussianDistribution = ref_vae.encode(composed_target).latent_dist
                    
                        ref_posterior, ref_posterior_black, ref_posterior_white = map(DiagonalGaussianDistribution, torch.chunk(ref_posterior.parameters, 3, dim=0))
                    
                    z = posterior.sample()
                    pred = unwrapped_vae.decode(z).sample
                    # if accelerator.is_local_main_process:
                    #     logger.info(f"z shape: {z.shape}")
                    
                    
                    l2_loss = loss_module.reconstruction_loss(pred, target)
                    loss = l2_loss
                    logs.update({"train/l2_loss": l2_loss})
                    
                    if args.lpips_scale is not None:
                        p_loss = loss_module.perceptual_loss(pred, target)
                        loss += args.lpips_scale * p_loss
                        logs.update({"train/p_loss": p_loss})
                    
                    if args.kl_scale is not None:
                        kl_norm_loss = loss_module.kl_loss(posterior)
                        loss += args.kl_scale * kl_norm_loss
                        logs.update({"train/kl_norm_loss": kl_norm_loss})
                    
                    if args.ref_kl_scale is not None:
                        kl_ref_white_loss = loss_module.kl_loss(posterior_white, ref_posterior_white)
                        kl_ref_black_loss = loss_module.kl_loss(posterior_black, ref_posterior_black)
                        kl_ref_loss = (kl_ref_black_loss + kl_ref_white_loss) / 2
                        loss += args.ref_kl_scale * kl_ref_loss
                        logs.update({"train/kl_ref_white_loss": kl_ref_white_loss})
                    
                    
                    if args.gan_start_step is not None and global_step >= args.gan_start_step:
                        loss_module.requires_grad_(False)
                        generator_loss = loss_module.generator_loss(l2_loss, pred, unwrapped_vae.decoder.conv_out.weight)
                        loss += generator_loss * args.generator_loss_weight
                        logs.update({"train/generator_loss": generator_loss})

                    logs.update({
                        "train/lr_generator": lr_scheduler_generator.get_last_lr()[0],
                        "train/loss": loss,
                    })
                    
                    # Update generator
                    optimizer_gen.zero_grad()
                    accelerator.backward(loss)
                    optimizer_gen.step()
                    lr_scheduler_generator.step()
                    
                    
                    discriminator_loss = 0
                    if args.gan_start_step is not None and global_step >= args.gan_start_step:
                        discriminator_loss = loss_module.discriminator_loss(target, pred)
                        logs.update({
                            "train/lr_discriminator": lr_scheduler_discriminator.get_last_lr()[0],
                            "train/discriminator_loss": discriminator_loss
                        })
                        optimizer_disc.zero_grad()
                        accelerator_discriminator.backward(discriminator_loss)
                        optimizer_disc.step()
                        lr_scheduler_discriminator.step()
                        
                    # del posterior, posterior_black, posterior_white, ref_posterior, ref_posterior_white, ref_posterior_black
                
            if accelerator.sync_gradients:
                if args.gan_start_step is not None:
                    if accelerator_discriminator.sync_gradients:
                        pass
                progress_bar.update(1)
                global_step += 1
                accelerator.log(logs, step=global_step)
                # train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.gan_start_step is not None:
                        accelerator_discriminator.save_state(os.path.join(save_path, "discriminator"))
                        
                    if accelerator.is_main_process:
                        hf_save_path = os.path.join(save_path, "hf")
                        unwrapped_vae.save_pretrained(hf_save_path)
                    logger.info(f"Saved state to {save_path}")
                        
                if accelerator.is_main_process and (global_step % args.validation_steps == 0 or (args.validation_on_first_step and global_step == 1)):
                    with torch.no_grad():
                        with accelerator.autocast():
                            log_validation(test_dataloader, vae, accelerator, weight_dtype, global_step)
    
            progress_bar.set_postfix({"lr": lr_scheduler_generator.get_last_lr()[0], "loss": loss.detach().item()})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
