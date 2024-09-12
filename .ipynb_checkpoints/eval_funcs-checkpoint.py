#pip install pytorch-fid


import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import warnings
import os
from pytorch_fid import fid_score
from fid_score_gpu import calculate_fid_given_paths

import shutil

import time
import subprocess

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./cin256-v2.yaml")
    model = load_model_from_config(config, "./model.ckpt")
    return model

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def save_samples_as_images(samples, folder_path, class_label, start_idx):
    class_folder = os.path.join(folder_path, f'{class_label}')
    
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    for i, sample in enumerate(samples):
        img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
        output_image = Image.fromarray(img.astype(np.uint8))
        # 배치 시작 인덱스와 배치 내 인덱스를 결합하여 파일 이름 생성
        output_image.save(os.path.join(class_folder, f'class_{class_label}_sample_{start_idx + i}.png'))

def sampling(model=None, output_folder = "output_samples", device="cuda", large_batch_size=4, small_batch_size=4, num_images=10000,cfg_scale=1.0, ddim_eta=1.0, DDIM_num_steps=25):
    
    if model is None:
        model = get_model()
    model.to(device)
    
    sampler = DDIMSampler(model)
    
    classes = list(range(1000))  # 샘플링할 클래스

    
    n_samples_per_class = num_images // len(classes)  # 클래스당 샘플 수
    
    ddim_steps = DDIM_num_steps # ddim step
    
    ddim_eta = ddim_eta  # eta 값
    
    scale = cfg_scale  # for unconditional guidance 0: uncond, 1: no guidance cond

    
    with torch.no_grad():
        # unet은 큰 배치 사이즈, vae는 작은 배치 사이즈 사용

        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(large_batch_size * [1000]).to(model.device)}
            )
            
            all_classes = []
            for class_label in classes:
                all_classes += [class_label] * n_samples_per_class
            
            for i in range(0, len(all_classes), large_batch_size):
                current_batch_size = min(large_batch_size, len(all_classes) - i)
                current_classes = all_classes[i:i + current_batch_size]
                
                xc = torch.tensor(current_classes)
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=current_batch_size,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 ddim_use_original_steps=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)
                
                torch.cuda.empty_cache()
                
                # 큰 배치로 샘플링 완료 후 작은 배치로 디코딩
                for j in range(0, current_batch_size, small_batch_size):
                    current_small_batch_size = min(small_batch_size, current_batch_size - j)
                    small_batch_samples = samples_ddim[j:j + current_small_batch_size]
                    
                    # 작은 배치 디코딩
                    x_samples_ddim = model.decode_first_stage(small_batch_samples)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                     # 배치 내 각 클래스에 맞는 폴더에 저장
                    for k, sample in enumerate(x_samples_ddim):
                        class_label = current_classes[j + k]
                        save_samples_as_images([sample], output_folder, class_label, i + j + k)

                
                # 메모리 캐시 정리
                torch.cuda.empty_cache()



def copy_files(source_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for root, _, files in os.walk(source_folder):
        for file in files:
            source_file = os.path.join(root, file)
            shutil.copy(source_file, dest_folder)

def copy_files_by_class_range(val_directory, start_class, end_class, destination):
    for class_index in range(start_class, end_class + 1):
        class_folder = os.path.join(val_directory, str(class_index))
        if os.path.exists(class_folder):
            copy_files(class_folder, destination)
        else:
            print(f"Class folder {class_folder} does not exist, skipping.")

def copy_files_to_all_destinations(val_directory, destination_0_to_949, destination_950_to_999, destination_all):
    # Step 1: Copy files from class 0 to 949
    copy_files_by_class_range(val_directory, 0, 949, destination_0_to_949)

    # Step 2: Copy files from class 950 to 999
    copy_files_by_class_range(val_directory, 950, 999, destination_950_to_999)

    # Step 3: Copy all files from class 0 to 999
    copy_files_by_class_range(val_directory, 0, 999, destination_all)

    print("File copying completed.")



def sample_and_cal_fid(device, num_images=10000, model=None, output_dir = "./output_samples/",ddim_eta=1.0, cfg_scale=1.0, DDIM_num_steps=25):

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()

    # Directories
    #output_dir = "./output_samples/"
    
    # sampling
    
    sampling(output_folder=output_dir, model=model, device=device, num_images=num_images, ddim_eta=ddim_eta, cfg_scale=cfg_scale, DDIM_num_steps=DDIM_num_steps)



    val_directory = output_dir

    destination_all = "./output_samples_all/"
    destination_0_to_949 = "./output_samples_0_to_949/"
    destination_950_to_999 = "./output_samples_950_to_999/"

    # Call the function to copy files to all destinations
    copy_files_to_all_destinations(val_directory, destination_0_to_949, destination_950_to_999, destination_all)


    
    '''
    gt_0_to_949: a (train data)
    gt_950_to_999: b (out-of-domain data)
    gy_all: c == (a + b)

    destination_0_to_949: a'
    destination_950_to_999: b'
    destination_all: c' == (a' + b')

    fid
    - (a,a')
    - (b,b')
    - (c, c')

    '''
    
    # faster-pytorch-fid -> https://github.com/jaywu109/faster-pytorch-fid/tree/main

    print("start fid_a")
    #calculate_fid("../imagenet_val_0_to_949.npz", destination_0_to_949, device)
    fid_value_a = calculate_fid_given_paths(["npz_files/imagenet_val_0_to_949.npz", destination_0_to_949], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048)
    
    print("start fid_b")
    fid_value_b = calculate_fid_given_paths(["npz_files/imagenet_val_950_to_999.npz", destination_950_to_999], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048)
    
    print("start fid_c")
    fid_value_c = calculate_fid_given_paths(["npz_files/imagenet_val_all.npz", destination_all], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048)






    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"FID score_a: {fid_value_a}")
    print(f"FID score_b: {fid_value_b}")
    print(f"FID score_c: {fid_value_c}")

    print(f"FID 실행 시간(sampling+cal_fid): {execution_time} 초")

    delete_folder(val_directory)
    delete_folder(destination_all)
    delete_folder(destination_950_to_999)
    delete_folder(destination_0_to_949)
    
    return fid_value_a, fid_value_b, fid_value_c


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = "./output_samples/"
    
    sample_and_cal_fid(device, output_dir)


if __name__ == "__main__":
    main()