import copy
import json
import os
import warnings
import argparse

import torch.nn.init as init
import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid, save_image
from diffusers.optimization import get_scheduler
from absl import app, args
from tqdm import trange
from tqdm import tqdm
import time
import wandb
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from trainer import distillation_DDPM_trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--scale_lr", type=bool, default=False, help="Flag to scale learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    
    parser.add_argument("--trainable_modules", type=tuple, default=(None,), help="Tuple of trainable modules")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer")

    return parser

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    model.eval()
    return model


def get_model_teacher():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def load_model_from_config_without_ckpt(config):
    print("Initializing model without checkpoint")
    model = instantiate_from_config(config.model)
    for param in model.parameters():
        param.requires_grad = True
    model.cuda()  # 모델을 CUDA로 이동 (필요한 경우)
    model.eval()  # 평가 모드로 설정
    return model

def get_model_student():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config_without_ckpt(config)
    return model

def initialize_params(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() > 1:  # Convolutional layers and Linear layers typically have more than 1 dimension
                    init.xavier_uniform_(param)
                else:
                    init.zeros_(param)



def distillation(args):

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        notes=args.wandb_notes,
        config={
            "learning_rate": args.lr,
            "architecture": "UNet",
            "dataset": "your_dataset_name",
            "epochs": args.total_steps,
        }
    )

    T_model = get_model_teacher()
    S_model= get_model_student()
    initialize_params(S_model)

    optimizer = torch.optim.AdamW(
        S_model,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    T_sampler = DDIMSampler(T_model)
    S_sampler = DDIMSampler(S_model)
    
    T_sampler.make_schedule(ddim_num_steps = args.DDIM_num_steps, ddim_eta= 1, verbose=False)
    S_sampler.make_schedule(ddim_num_steps = args.DDIM_num_steps, ddim_eta= 1, verbose=False)
     
    trainer = distillation_DDPM_trainer(
        T_model, S_model, T_sampler, S_sampler, args.train_is_feature, args.beta_1, args.beta_T, args.T,
        args.mean_type, args.var_type, args.distill_features).to(device)
    

    cache_size = args.cache_n*1000
    
    img_cache = torch.randn(cache_size, T_model.channels, T_model.img_size, T_model.img_size).to(device)
    t_cache = torch.ones(cache_size, dtype=torch.long, device=device)*(T_model.timestep-1)
    class_cache = torch.randint(0, 950, (cache_size,), device=device)

    # 10%의 인덱스를 무작위로 선택하여 1000으로 설정
    num_to_replace = int(cache_size * 0.1)  # 전체 크기의 10%
    indices = torch.randperm(cache_size)[:num_to_replace]  # 랜덤으로 인덱스 선택
    class_cache[indices] = 1000

    with torch.no_grad():
        for i in range(0, cache_size, args.batch_size):
            # 슬라이스의 끝 인덱스가 전체 크기를 초과하지 않도록 min 사용
            end_idx = min(i + args.batch_size, cache_size)
            
            # 슬라이스 처리
            img_batch = img_cache[i:end_idx]
            t_batch = t_cache[i:end_idx]
            class_batch = class_cache[i:end_idx]
            
            uc = T_model.get_learned_conditioning(
                        {T_model.cond_stage_key: torch.tensor(img_batch.shape[0] * [1000]).to(device)}
                    )
            
            c = T_model.get_learned_conditioning(
                        {T_model.cond_stage_key: class_batch})

            img_cache[i:end_idx], _ = T_sampler.X0_DDPM(S=args.ddim_steps,
                                            conditioning=c,
                                            ddim_use_original_steps = True, #True  #### DDIM for cache??
                                            batch_size=img_batch.shape[0],
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=args.CFG_scale, #우선 1로
                                            unconditional_conditioning=uc,
                                            eta=1)
                    
    ##################################

    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # Step 2: Randomly sample from img_cache and t_cache without shuffling
            indices = torch.randint(0, img_cache.size(0), (args.batch_size,), device=device)

            # Sample img_cache and t_cache using the random indices
            x_t = img_cache[indices]
            t = t_cache[indices]
            c = T_model.get_learned_conditioning(
                        {T_model.cond_stage_key: class_cache[indices]})
            
            # Calculate distillation loss
            output_loss, x_t_, total_loss = trainer(x_t, c, t, args.CFG_scale)

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(S_model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            ### cache update ###
            img_cache[indices] = x_t_
            t_cache[indices] -= 1
            
            num_999 = torch.sum(t_cache == (args.T - 1)).item()

            if num_999 < args.cache_n:
                missing_999 = args.cache_n - num_999
                non_999_indices = (t_cache != (args.T - 1)).nonzero(as_tuple=True)[0]
                random_indices = torch.randperm(non_999_indices.size(0), device=device)[:missing_999]
                selected_indices = non_999_indices[random_indices]
                t_cache[selected_indices] = args.T - 1
                img_cache[selected_indices] = torch.randn(missing_999, 3, args.img_size, args.img_size, device=device)

            # t_cache에서 값이 0인 인덱스를 찾아 초기화
            zero_indices = (t_cache < 0).nonzero(as_tuple=True)[0]
            num_zero_indices = zero_indices.size(0)

            # 0인 인덱스가 있는 경우에만 초기화 수행
            if num_zero_indices > 0:
                # 0인 인덱스를 1에서 args.T-1 사이의 랜덤한 정수로 초기화
                t_cache[zero_indices] = torch.randint(0, args.T, size=(num_zero_indices,), dtype=torch.long, device=device)
                img_cache[zero_indices] = trainer.diffusion(img_cache[zero_indices],t_cache[zero_indices])


            # Logging with WandB
            wandb.log({
                'distill_loss': total_loss.item(),
                'output_loss': output_loss.item()
                       }, step=step)
            pbar.set_postfix(distill_loss='%.3f' % total_loss.item())
             
            ################### Sample and save student outputs############################

            ################### Save student model ################################

            ################### Evaluate student model ##############################

    wandb.finish()


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # distill_caching_base()
    parser = get_parser()
    distill_args = parser.parse_args()
    seed_everything(distill_args.seed)
    distillation(distill_args)

if __name__ == '__main__':
    app.run(main)