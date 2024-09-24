import os
import random
import torch
import torch.nn.init as init
from torchvision.utils import make_grid
import numpy as np
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist
import shutil
import time


from ldm.util import instantiate_from_config

def visualize_t_cache_distribution(t_cache, cache_n):
    # CPU로 이동하여 numpy 배열로 변환
    t_cache_cpu = t_cache.cpu().numpy()

    # 히스토그램을 그려 분포 확인 (bin 수를 1000으로 설정)
    plt.figure(figsize=(12, 6))
    plt.hist(t_cache_cpu, range=(0, 1000), bins=1000, alpha=0.7, color='blue')
    plt.title('Distribution of t_cache')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.ylim(0, cache_n*10)
    plt.grid(True)

    # 저장할 디렉토리가 없다면 생성
    os.makedirs('./cache_test', exist_ok=True)

    plt.savefig('./cache_test/temp_frame.png')
    plt.close()


def print_gpu_memory_usage(step_description):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        print(f"[{step_description}] GPU Memory Allocated: {allocated_memory / (1024 ** 2):.2f} MB")
        print(f"[{step_description}] GPU Memory Reserved: {reserved_memory / (1024 ** 2):.2f} MB")

    else:
        print("CUDA is not available.")

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    for param in model.parameters():
        param.requires_grad = False
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
    return model

def load_model_from_config_with_ckpt(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    for param in model.parameters():
        param.requires_grad = True
    return model

def get_model_student():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config_with_ckpt(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def get_model(config_path):
    config = OmegaConf.load(config_path)
    model = load_model_from_config_without_ckpt(config)
    return model
##########################################################################################################################3
def load_model_from_config_with_ckpt2(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cpu')  # Load the checkpoint
    
    if "student_model" in pl_sd:
        sd = pl_sd["student_model"]  # 정확하게 student_model의 state_dict를 가져옴
    else:
        raise KeyError("The checkpoint does not contain the expected 'student_model' key.")
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model

def get_model_student_with_ckpt():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config_with_ckpt2(config, "logs/cin256-v2/1/student_ckpt_step_100000.pt")
    #model = load_model_from_config_with_ckpt(config, "models/ldm/cin256-v2/model.ckpt")
    return model
##########################################################################################################################3


def initialize_params(model):
    target_model = model.model  # S_model의 .model 부분에 접근
    
    for name, param in target_model.named_parameters():
        if param.requires_grad:
            if param.dim() > 1:  # Convolutional layers and Linear layers typically have more than 1 dimension
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
                
    for name, param in model.named_parameters():
        if 'model' not in name:  # model 파라미터가 아닌 경우
            param.requires_grad = False
            
def get_small_model_student():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2_small.yaml")
    model = load_model_from_config_without_ckpt(config)
    #model = load_model_from_config_with_ckpt(config, "models/ldm/cin256-v2/model.ckpt")
    return model


def copy_weight_from_teacher(unet_stu, unet_tea):

    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    
    connect_info['model.diffusion_model.input_blocks.2.'] = 'model.diffusion_model.input_blocks.3.'
    connect_info['model.diffusion_model.input_blocks.3.'] = 'model.diffusion_model.input_blocks.4.'
    connect_info['model.diffusion_model.input_blocks.4.'] = 'model.diffusion_model.input_blocks.6.'
    connect_info['model.diffusion_model.input_blocks.5.'] = 'model.diffusion_model.input_blocks.7.'
    connect_info['model.diffusion_model.input_blocks.6.'] = 'model.diffusion_model.input_blocks.9.'
    connect_info['model.diffusion_model.input_blocks.7.'] = 'model.diffusion_model.input_blocks.10.'

    connect_info['model.diffusion_model.output_blocks.1.'] = 'model.diffusion_model.output_blocks.2.'
    connect_info['model.diffusion_model.output_blocks.2.'] = 'model.diffusion_model.output_blocks.3.'
    connect_info['model.diffusion_model.output_blocks.3.'] = 'model.diffusion_model.output_blocks.5.'
    connect_info['model.diffusion_model.output_blocks.4.'] = 'model.diffusion_model.output_blocks.6.'
    connect_info['model.diffusion_model.output_blocks.5.'] = 'model.diffusion_model.output_blocks.8.'
    connect_info['model.diffusion_model.output_blocks.6.'] = 'model.diffusion_model.output_blocks.9.'
    connect_info['model.diffusion_model.output_blocks.7.'] = 'model.diffusion_model.output_blocks.11.'
        
    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])            
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
            

        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu

def copy_first_cond_weight_from_teacher(unet_stu, unet_tea):
    
    for k in unet_stu.state_dict().keys():
        # Skip parameters that start with 'model.diffusion_model.'
        if k.startswith('model.diffusion_model.'):
            continue            

        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k])
        
        print(f"Student COPY {k}")
    return unet_stu    


def save_checkpoint(S_model, optimizer, step, logdir):
    ckpt = {
        'student_model': S_model.state_dict(),  # S_model의 상태 저장
        #'scheduler': lr_scheduler.state_dict(),  # 학습 스케줄러 상태 저장
        'optimizer': optimizer.state_dict(),  # 옵티마이저 상태 저장
        'step': step,  # 현재 스텝 저장
    }
    
    # 저장 경로 설정
    save_path = os.path.join(logdir, f'student_ckpt_step_{step}.pt')
    
    # 디렉터리 존재 여부 확인 후 생성
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    
    # 체크포인트 저장
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved at step {step} to {save_path}")

def save_cache(cache_dataset, step, logdir, rank):
    
    cache_dir = os.path.join(logdir, f'cache_step_{step}')
    cache_save_path = os.path.join(cache_dir, f'cache_step_{step}_rank_{rank}.pt')
    
    # 디렉토리 존재 여부를 확인하고, 없으면 생성
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'img_cache': cache_dataset.img_cache,
        't_cache': cache_dataset.t_cache,
        'c_emb_cache': cache_dataset.c_emb_cache,
        'class_cache': cache_dataset.class_cache,
    }
    
    torch.save(cache_data, cache_save_path)
    print(f"Cache saved at step {step} to {cache_save_path}")
    
    
def sample_save_images(num_sample_class, n_sample_per_class, steps, DDPM_sampling, eta,
                       cfg_scale, T_model, S_model, T_sampler, S_sampler, step, rank, world_size):
    classes_per_gpu = num_sample_class // world_size
    remainder_classes = num_sample_class % world_size
    start_class = rank * classes_per_gpu + min(rank, remainder_classes)
    end_class = start_class + classes_per_gpu + (1 if rank < remainder_classes else 0)
    classes = [390, 29, 505, 379, 465, 133, 116, 657]
    classes = classes[start_class:end_class]
    n_samples_per_class = n_sample_per_class
    ddim_steps = steps
    ddim_eta = eta
    scale = cfg_scale # for unconditional guidance
    all_samples_T = list()  # T_model 샘플 저장
    all_samples_S = list()  # S_model 샘플 저장
    T_model = T_model
    S_model = S_model
    if DDPM_sampling:
        noises = torch.randn(1001, len(classes)*n_samples_per_class, 3, 64, 64, device=T_model.device)
    else:
        noises = torch.randn(1+ddim_steps, len(classes)*n_samples_per_class, 3, 64, 64, device=T_model.device)
    with torch.no_grad():
        for model_name, model, sampler in zip(["T_model", "S_model"], [T_model, S_model], [T_sampler, S_sampler]):
            model.eval()
            with model.ema_scope():
                # print("##########################################################################################")
                # print(f"device: {rank}, seed: {[x for x in torch.get_rng_state().tolist() if x != 0][:10]}")
                # print("##########################################################################################")
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
                )
                for i, class_label in enumerate(classes):
                    print(f"rendering {n_samples_per_class} examples of class '{class_label}' using {model_name} in {ddim_steps} steps and using s={scale:.2f}.")
                    xc = torch.tensor(n_samples_per_class * [class_label])
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                    samples_ddim, _ = sampler.sample_with_noise(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=n_samples_per_class,
                                                     shape=[3, 64, 64],
                                                     noises = noises[:,i*n_samples_per_class:(i+1)*n_samples_per_class,:,:,:],
                                                     verbose=False,
                                                     ddim_use_original_steps=DDPM_sampling,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    if model_name == "T_model":
                        all_samples_T.append(x_samples_ddim)
                    else:
                        all_samples_S.append(x_samples_ddim)
    # 이미지 그리드 생성
    grid_T = torch.stack(all_samples_T, 0)
    grid_T = rearrange(grid_T, 'n b c h w -> (n b) c h w')
    grid_T = make_grid(grid_T, nrow=n_samples_per_class)
    grid_S = torch.stack(all_samples_S, 0)
    grid_S = rearrange(grid_S, 'n b c h w -> (n b) c h w')
    grid_S = make_grid(grid_S, nrow=n_samples_per_class)
    # 저장할 디렉토리 경로 설정
    save_dir = "save_images"
    # 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # 각 GPU에서 저장할 파일 경로
    output_image_T_path = os.path.join(save_dir, f'output_T_model_{rank}.png')
    output_image_S_path = os.path.join(save_dir, f'output_S_model_{rank}.png')
    # 이미지 데이터를 처리하고 저장
    grid_T = 255. * rearrange(grid_T, 'c h w -> h w c').cpu().numpy()
    grid_S = 255. * rearrange(grid_S, 'c h w -> h w c').cpu().numpy()
    output_image_T = Image.fromarray(grid_T.astype(np.uint8))
    output_image_S = Image.fromarray(grid_S.astype(np.uint8))
    # 이미지 저장
    output_image_T.save(output_image_T_path)
    output_image_S.save(output_image_S_path)
    # rank 0에서 이미지를 모아서 하나로 합침
    dist.barrier()  # 모든 GPU가 작업을 완료할 때까지 대기
    if rank == 0:
        images_T = [Image.open(f'{save_dir}/output_T_model_{i}.png') for i in range(world_size)]
        images_S = [Image.open(f'{save_dir}/output_S_model_{i}.png') for i in range(world_size)]
        # 이미지를 하나로 합침 (단순하게 세로로 연결)
        total_image_T = np.concatenate([np.array(img) for img in images_T], axis=0)
        total_image_S = np.concatenate([np.array(img) for img in images_S], axis=0)
        # 합쳐진 이미지를 다시 저장
        final_image_T = Image.fromarray(total_image_T)
        final_image_S = Image.fromarray(total_image_S)
        final_image_T_path = 'final_output_T_model.png'
        final_image_S_path = 'final_output_S_model.png'
        final_image_T.save(final_image_T_path)
        final_image_S.save(final_image_S_path)
        # WandB에 로그 (합쳐진 이미지만 업로드)
        wandb.log({"Teacher Sample": wandb.Image(final_image_T_path)}, step=int(step))
        wandb.log({"Student Sample": wandb.Image(final_image_S_path)}, step=int(step))

        
        # # 저장한 이미지 삭제
        # try:
        #     shutil.rmtree(save_dir)
        #     print(f"The directory '{save_dir}' and all its contents have been successfully deleted.")
        # except Exception as e:
        #     print(f"Error deleting directory '{save_dir}': {e}")
        print("Combined T_model and S_model outputs have been saved and logged to WandB.")
        
    dist.barrier()


    
# def sample_save_images(num_sample_class, n_sample_per_class, steps, DDPM_sampling, eta,
#                        cfg_scale, T_model, S_model, T_sampler, S_sampler, step, rank, world_size, seed=None, selected_tensor=None):
    
#     if seed is not None:
#         set_seed(seed)  # 시드 고정
    
#     classes_per_gpu = num_sample_class // world_size
#     remainder_classes = num_sample_class % world_size

#     start_class = rank * classes_per_gpu + min(rank, remainder_classes)
#     end_class = start_class + classes_per_gpu + (1 if rank < remainder_classes else 0)
#     # print(f"Classes per GPU: {classes_per_gpu}, Remainder classes: {remainder_classes}")
#     # print(f"Start class: {start_class}, End class: {end_class}")
    
#     # # 이미 선택된 텐서에서 2개의 클래스를 무작위로 선택
#     # selected_classes = random.sample(selected_tensor.tolist(), 4)

#     # # 선택되지 않은 클래스들 중에서 2개의 클래스를 무작위로 선택
#     # all_classes = set(range(1000))  # 0부터 999까지의 모든 클래스
#     # selected_set = set(selected_tensor.tolist())  # 선택된 클래스들의 집합
#     # remaining_classes = list(all_classes - selected_set)  # 선택되지 않은 클래스들
#     # unseen_sampled = random.sample(remaining_classes, 4)

#     # # 최종적으로 4개의 클래스 리스트 만들기 (선택된 2개, 선택되지 않은 2개)
#     # classes = selected_classes + unseen_sampled    
#     classes = [390, 29, 505, 379, 465, 133, 116, 657]

#     classes = classes[start_class:end_class]
#     n_samples_per_class = n_sample_per_class
#     ddim_steps = steps
#     ddim_use_original_steps = (steps == 1000)
#     ddim_eta = eta
#     scale = cfg_scale  # for unconditional guidance
#     all_samples_T = list()  # T_model 샘플 저장
#     all_samples_S = list()  # S_model 샘플 저장
#     # print("#######################################################################")
#     # print("classes: ", classes)
#     # print("#######################################################################")

#     with torch.no_grad():
#         for model_name, model, sampler in zip(["T_model", "S_model"], [T_model, S_model], [T_sampler, S_sampler]):
#             model.eval()
#             with model.ema_scope():
#                 uc = model.get_learned_conditioning(
#                     {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
#                 )
#                 for class_label in classes:
#                     xc = torch.tensor(n_samples_per_class * [class_label])
#                     c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
#                     samples_ddim, _ = sampler.sample(S=ddim_steps,
#                                                      conditioning=c,
#                                                      batch_size=n_samples_per_class,
#                                                      shape=[3, 64, 64],
#                                                      verbose=False,
#                                                      ddim_use_original_steps=DDPM_sampling,
#                                                      unconditional_guidance_scale=scale,
#                                                      unconditional_conditioning=uc,
#                                                      eta=ddim_eta)
#                     x_samples_ddim = model.decode_first_stage(samples_ddim)
#                     x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
#                     if model_name == "T_model":
#                         all_samples_T.append(x_samples_ddim)
#                     else:
#                         all_samples_S.append(x_samples_ddim)

#     # 이미지 그리드 생성
#     grid_T = torch.stack(all_samples_T, 0)
#     grid_T = rearrange(grid_T, 'n b c h w -> (n b) c h w')
#     grid_T = make_grid(grid_T, nrow=n_samples_per_class)

#     grid_S = torch.stack(all_samples_S, 0)
#     grid_S = rearrange(grid_S, 'n b c h w -> (n b) c h w')
#     grid_S = make_grid(grid_S, nrow=n_samples_per_class)
    

#     # 저장할 디렉토리 경로 설정
#     save_dir = "save_images"

#     # 폴더가 없으면 생성
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 각 GPU에서 저장할 파일 경로
#     output_image_T_path = os.path.join(save_dir, f'output_T_model_{rank}.png')
#     output_image_S_path = os.path.join(save_dir, f'output_S_model_{rank}.png')

#     # 이미지 데이터를 처리하고 저장
#     grid_T = 255. * rearrange(grid_T, 'c h w -> h w c').cpu().numpy()
#     grid_S = 255. * rearrange(grid_S, 'c h w -> h w c').cpu().numpy()

#     output_image_T = Image.fromarray(grid_T.astype(np.uint8))
#     output_image_S = Image.fromarray(grid_S.astype(np.uint8))

#     # 이미지 저장
#     output_image_T.save(output_image_T_path)
#     output_image_S.save(output_image_S_path)

#     # rank 0에서 이미지를 모아서 하나로 합침
#     dist.barrier()  # 모든 GPU가 작업을 완료할 때까지 대기
#     if rank == 0:
#         images_T = [Image.open(f'{save_dir}/output_T_model_{i}.png') for i in range(world_size)]
#         images_S = [Image.open(f'{save_dir}/output_S_model_{i}.png') for i in range(world_size)]

#         # 이미지를 하나로 합침 (단순하게 세로로 연결)
#         total_image_T = np.concatenate([np.array(img) for img in images_T], axis=0)
#         total_image_S = np.concatenate([np.array(img) for img in images_S], axis=0)

#         # 합쳐진 이미지를 다시 저장
#         final_image_T = Image.fromarray(total_image_T)
#         final_image_S = Image.fromarray(total_image_S)
        
#         final_image_T_path = 'final_output_T_model.png'
#         final_image_S_path = 'final_output_S_model.png'

#         final_image_T.save(final_image_T_path)
#         final_image_S.save(final_image_S_path)

#         # WandB에 로그 (합쳐진 이미지만 업로드)
#         wandb.log({"Teacher Sample": wandb.Image(final_image_T_path)}, step=int(step))
#         wandb.log({"Student Sample": wandb.Image(final_image_S_path)}, step=int(step))
        
#         # 저장한 이미지 삭제
#         os.remove(output_image_T_path)
#         os.remove(output_image_S_path)

#         print("Combined T_model and S_model outputs have been saved and logged to WandB.")
    
#     dist.barrier()  # 다시 모든 GPU 동기화
    