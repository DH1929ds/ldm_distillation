import os
import random
import torch
import torch.nn.init as init
from torchvision.utils import make_grid
import numpy as np
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
import wandb

from ldm.util import instantiate_from_config



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
                    
def save_checkpoint(S_model, lr_scheduler, optimizer, step, logdir):
    ckpt = {
        'student_model': S_model.state_dict(),  # S_model의 상태 저장
        'scheduler': lr_scheduler.state_dict(),  # 학습 스케줄러 상태 저장
        'optimizer': optimizer.state_dict(),  # 옵티마이저 상태 저장
        'step': step,  # 현재 스텝 저장
    }
    # 저장 경로 설정
    save_path = os.path.join(logdir, f'student_ckpt_step_{step}.pt')
    # 체크포인트 저장
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved at step {step} to {save_path}")
    
def sample_save_images(num_sample_class, n_sample_per_class, steps, eta, cfg_scale, T_model, S_model, S_sampler, step):
    classes = random.sample(range(1000), num_sample_class)
    n_samples_per_class = n_sample_per_class
    ddim_steps = steps
    ddim_eta = eta
    scale = cfg_scale # for unconditional guidance
    all_samples_T = list()  # T_model 샘플 저장
    all_samples_S = list()  # S_model 샘플 저장
    T_model = T_model
    S_model = S_model
    with torch.no_grad():
        for model_name, model in zip(["T_model", "S_model"], [T_model, S_model]):
            model.eval()
            with model.ema_scope():
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
                )
                for class_label in classes:
                    print(f"rendering {n_samples_per_class} examples of class '{class_label}' using {model_name} in {ddim_steps} steps and using s={scale:.2f}.")
                    xc = torch.tensor(n_samples_per_class * [class_label])
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                    samples_ddim, _ = S_sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=n_samples_per_class,
                                                     shape=[3, 64, 64],
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    if model_name == "T_model":
                        all_samples_T.append(x_samples_ddim)
                    else:
                        all_samples_S.append(x_samples_ddim)
    # T_model 결과를 그리드로 변환 및 저장
    grid_T = torch.stack(all_samples_T, 0)
    grid_T = rearrange(grid_T, 'n b c h w -> (n b) c h w')
    grid_T = make_grid(grid_T, nrow=n_samples_per_class)
    # S_model 결과를 그리드로 변환 및 저장
    grid_S = torch.stack(all_samples_S, 0)
    grid_S = rearrange(grid_S, 'n b c h w -> (n b) c h w')
    grid_S = make_grid(grid_S, nrow=n_samples_per_class)
    # 각각의 그리드를 이미지로 변환
    grid_T = 255. * rearrange(grid_T, 'c h w -> h w c').cpu().numpy()
    grid_S = 255. * rearrange(grid_S, 'c h w -> h w c').cpu().numpy()
    # 이미지로 저장 (T_model과 S_model의 결과)
    output_image_T = Image.fromarray(grid_T.astype(np.uint8))
    output_image_T_path = 'output_T_model.png'
    output_image_T.save(output_image_T_path)
    output_image_S = Image.fromarray(grid_S.astype(np.uint8))
    output_image_S_path = 'output_S_model.png'
    output_image_S.save(output_image_S_path)
    # WandB에 각각의 이미지 로깅
    wandb.log({"Teacher Sample": wandb.Image(output_image_T_path)}, step=step)
    wandb.log({"Student Sample": wandb.Image(output_image_S_path)}, step=step)
    print("T_model and S_model outputs have been saved and logged to WandB.")
                           