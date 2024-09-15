import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

import warnings
import random
import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import warnings
import threading
import diffusers
import argparse, os, sys, glob, yaml, math, random
import time
from itertools import cycle

from gpu_log import GPUMonitor
from tqdm import trange
from pytorch_lightning import seed_everything
from diffusers.optimization import get_scheduler

from ldm.models.diffusion.ddim import DDIMSampler
from trainer import distillation_DDPM_trainer
from funcs import load_model_from_config, get_model_teacher, load_model_from_config_without_ckpt, get_model_student, initialize_params, sample_save_images, save_checkpoint, print_gpu_memory_usage, visualize_t_cache_distribution
from eval_funcs import sample_and_cal_fid
from data_loaders.cache_data import load_cache, Cache_Dataset, custom_collate_fn

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=20240911, help="seed for seed_everything")
    parser.add_argument('--world_size', type=int, default=None, help='number of GPU to use for training')
    parser.add_argument("--MASTER_PORT", type=str, default="12355", help="MASTER_PORT for DDP")
    parser.add_argument("--MASTER_ADDR", type=str, default="127.0.0.1", help="MASTER_ADDR fof DDP")
    
    parser.add_argument("--trainable_modules", type=tuple, default=(None,), help="Tuple of trainable modules")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon parameter for Adam optimizer")

    # Gaussian Diffusion
    parser.add_argument("--beta_1", type=float, default=1e-4, help='start beta value')
    parser.add_argument("--beta_T", type=float, default=0.02, help='end beta value')
    parser.add_argument("--T", type=int, default=1000, help='total diffusion steps')
    parser.add_argument("--mean_type", type=str, choices=['xprev', 'xstart', 'epsilon'], default='epsilon', help='predict variable')
    parser.add_argument("--var_type", type=str, choices=['fixedlarge', 'fixedsmall'], default='fixedlarge', help='variance type')
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--scale_lr", type=bool, default=False, help="Flag to scale learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1., help="gradient norm clipping")
    parser.add_argument("--total_steps", type=int, default=800000, help='total training steps')
    parser.add_argument("--epoch", type=int, default=800000, help='epoch')
    parser.add_argument("--warmup", type=int, default=5000, help='learning rate warmup')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--num_workers", type=int, default=4, help='workers of Dataloader')
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="ema decay rate")
    parser.add_argument("--parallel", action='store_true', help='multi gpu training')
    parser.add_argument("--distill_features", action='store_true', help='perform knowledge distillation using intermediate features')
    parser.add_argument("--loss_weight", type=float, default=0.1, help="feature loss weighting")
    
    # Logging & Sampling
    parser.add_argument("--logdir", type=str, default='./logs/cin256-v2', help='log directory')
    parser.add_argument("--sample_size", type=int, default=32, help="sampling size of images")
    parser.add_argument("--sample_step", type=int, default=10000, help='frequency of sampling')
    
    # WandB 관련 FLAGS 추가
    parser.add_argument("--wandb_project", type=str, default='distill_caching_ldm', help='WandB project name')
    parser.add_argument("--wandb_run_name", type=str, default=None, help='WandB run name')
    parser.add_argument("--wandb_notes", type=str, default='', help='Notes for the WandB run')
    
    # Evaluation
    parser.add_argument("--save_step", type=int, default=50000, help='frequency of saving checkpoints, 0 to disable during training')
    parser.add_argument("--eval_step", type=int, default=0, help='frequency of evaluating model, 0 to disable during training')
    parser.add_argument("--num_images", type=int, default=10000, help='the number of generated images for evaluation')
    parser.add_argument("--fid_use_torch", action='store_true', help='calculate IS and FID on gpu')
    parser.add_argument("--fid_cache", type=str, default='./stats/cifar10.train.npz', help='FID cache')
    
    # Caching
    parser.add_argument("--pre_caching", action='store_true', help='only precaching')
    parser.add_argument("--cache_n", type=int, default=64, help='size of caching data per timestep')
    parser.add_argument("--caching_batch_size", type=int, default=256, help='batch size for pre-caching')
    parser.add_argument('--cachedir', type=str, default='./cache', help='log directory')

    #DDIM Sampling
    parser.add_argument("--DDPM_sampling", action="store_true", help="sampling using DDPM sampling")
    parser.add_argument("--DDIM_num_steps", type=int, default=50, help='number of DDIM samping steps')
    parser.add_argument("--num_sample_class", type=int, default=4, help='number of class for save and sampling')
    parser.add_argument("--n_sample_per_class", type=int, default=16, help='number of sample for per class in save_sample')
    parser.add_argument("--sample_save_ddim_steps", type=int, default=20, help='number of DDIM sampling steps')
    parser.add_argument("--ddim_eta", type=float, default=1.0, help='DDIM eta parameter for noise level')
    parser.add_argument("--cfg_scale", type=float, default=1, help='guidance scale for unconditional guidance, 1 or none = no guidance, 0 = uncond')


    return parser

def pre_caching(args):

    device = torch.device('cuda:0')
    #gpu_monitor = GPUMonitor(monitoring_int
    
 
    #gpu_monitor.start("model_load_start!!")

    T_model = get_model_teacher()
    T_model = T_model.to(device)

    T_device = T_model.device

    T_sampler = DDIMSampler(T_model)
    #print_gpu_memory_usage('models to sampler')
    
    T_sampler.make_schedule(ddim_num_steps = args.DDIM_num_steps, ddim_eta= 1, verbose=False)

    ############################################ precacheing ##################################################
    cache_size = args.cache_n*args.T
    
    img_cache = torch.randn(cache_size, T_model.channels, T_model.image_size, T_model.image_size).to(T_device)
    t_cache = torch.ones(cache_size, dtype=torch.long, device=T_device)*(args.T-1)
    
    selected_tensor  = torch.tensor(
        [862, 43, 335, 146, 494, 491, 587, 588, 187, 961, 78, 205, 297, 214, 163, 788, 980, 507, 916, 112, 512, 589, 771, 27, 269, 386, 336, 280, 362, 510, 850, 661, 731, 613, 945, 704, 86, 160, 372, 910, 159, 493, 623, 73, 128, 234, 717, 710, 887, 423, 546, 148, 558, 358, 463, 224, 987, 960, 444, 965, 363, 854, 492, 87, 672, 870, 217, 292, 303, 508, 188, 296, 642, 349, 154, 690, 298, 670, 964, 341, 873, 236, 35, 28, 890, 698, 902, 457, 621, 629, 371, 114, 610, 186, 718, 815, 944, 832, 869, 919, 441, 394, 625, 993, 401, 650, 55, 825, 272, 233, 738, 483, 473, 8, 220, 547, 684, 533, 132, 646, 455, 895, 52, 400, 593, 943, 848, 380, 175, 951, 195, 404, 856, 464, 123, 10, 433, 283, 366, 122, 307, 460, 616, 585, 407, 785, 835, 712, 912, 397, 440, 901, 600, 732, 140, 499, 864, 653, 584, 844, 874, 420, 147, 574, 24, 183, 243, 379, 338, 699, 94, 79, 254, 458, 430, 350, 388, 711, 639, 415, 299, 412, 743, 340, 967, 17, 992, 480, 858, 393, 918, 193, 334, 324, 575, 130, 950, 759, 820, 244, 652, 171, 18, 576, 15, 581, 93, 290, 847, 505, 922, 883, 470, 293, 777, 696, 215, 322, 291, 540, 416, 40, 956, 488, 780, 184, 453, 792, 127, 200, 602, 378, 344, 273, 255, 935, 763, 714, 529, 700, 226, 76, 502, 566, 165, 106, 867, 811, 376, 802, 678, 267, 276, 767, 881, 248, 26, 567, 995, 143, 709, 124, 927, 431, 270, 29, 966, 926, 168, 769, 149, 786, 761, 14, 22, 474, 981, 257, 676, 662, 96, 872, 679, 177, 413, 928, 314, 185, 120, 687, 395, 599, 346, 737, 352, 638, 157, 716, 974, 783, 467, 697, 559, 181, 797, 111, 144, 389, 834, 715, 894, 70, 206, 666, 0, 190, 520, 142, 259, 429, 948, 729, 841, 830, 764, 232, 150, 446, 80, 782, 225, 391, 477, 720, 295, 319, 803, 182, 989, 831, 800, 166, 506, 563, 721, 135, 305, 904, 145, 427, 72, 178, 947, 975, 33, 706, 997, 60, 828, 829, 45, 432, 482, 98, 392, 846, 968, 381, 577, 57, 240, 179, 484, 167, 282, 969, 542, 768, 930, 65, 239, 359, 107, 619, 218, 824, 503, 733, 515, 958, 469, 288, 606, 439, 622, 618, 419, 971, 294, 263, 504, 247, 744, 651, 310, 806, 339, 434, 633, 204, 659, 702, 351, 85, 81, 673, 449, 591, 537, 572, 668, 227, 580, 655, 962, 724, 937, 766, 742, 194, 285, 435, 897, 462, 708, 776, 693, 192, 582, 843, 597, 437, 513, 357, 365, 398, 713, 990, 523, 946, 837, 840, 564, 608, 855, 522, 719, 849, 603, 853, 691, 550, 37, 809, 778, 89, 321, 548, 309, 102, 41, 745, 399, 631, 7, 812, 421, 554, 119, 472, 438, 32, 481, 685, 817, 490, 723, 12, 570, 9, 568, 387, 164, 211, 6, 46, 448, 695, 242, 521, 978, 814, 875, 607, 634, 931, 884, 614, 320, 251, 77, 237, 118, 810, 617, 61, 311, 703, 963, 772, 972, 878, 571, 794, 868, 67, 774, 674, 976, 955, 49, 842, 117, 216, 932, 632, 134, 109, 994, 308, 747, 245, 517, 991, 648, 249, 643, 628, 590, 90, 30, 279, 345, 770, 544, 795, 705, 126, 913, 936, 636, 985, 219, 497, 751, 383, 410, 20, 63, 424, 138, 230, 261, 235, 649, 13, 1, 929, 228, 906, 38, 560, 598, 436, 798, 375, 921, 396, 5, 354, 640, 83, 3, 624, 511, 725, 630, 826, 333, 425, 361, 411, 626, 773, 471, 556, 728, 781, 161, 278, 790, 601, 450, 384, 996, 317, 565, 489, 804, 755, 641, 4, 277, 405, 539, 819, 115, 892, 113, 551, 734, 527, 924, 325, 451, 957, 367, 342, 323, 973, 289, 356, 327, 23, 545, 941, 36, 534, 784, 11, 977, 671, 637, 536, 905, 821, 669, 369, 101, 748, 907, 370, 212, 108, 579, 920, 595, 377, 31, 390, 409, 137, 189, 2, 222, 983, 667, 557, 385, 201, 970, 586, 692, 287, 866, 796, 58, 238, 726, 984, 445, 258, 75, 92, 355, 665, 153, 300, 162, 675, 596, 208, 903, 500, 466, 442, 286, 838, 328, 54, 531, 34, 456, 877, 689, 97, 552, 891, 760, 543, 199, 418, 660, 459, 100, 19, 514, 274, 246, 681, 647, 253, 805, 645, 900, 765, 938, 752, 50, 663, 151, 683, 526, 461, 741, 917, 152, 88, 301, 68, 125, 203, 498, 275, 822, 934, 654, 56, 155, 110, 735, 131, 74, 156, 262, 443, 207, 871, 525, 818, 306, 562, 173, 454, 485, 739, 382, 677, 364, 501, 942, 402, 749, 914, 172, 813, 176, 865, 573, 753, 592, 827, 62, 48, 347, 284, 852, 82, 730, 816, 71, 518, 909, 213, 953, 21, 688, 202, 360, 882, 141, 69, 105, 898, 104, 611, 315, 403, 999, 561, 374, 694, 876, 252, 496, 754, 158, 84, 808, 982, 532, 281, 42, 264, 348, 896, 174, 373, 879, 845, 911, 406, 198, 422, 583, 59, 535, 680, 627, 923, 316, 51, 265, 620, 417, 549, 353, 727, 304, 495, 250, 475, 241, 538, 312, 682, 66, 95, 658, 426, 266, 479, 578, 889, 368, 476, 988, 801, 740, 886, 807, 787, 414, 197, 940, 210, 779, 452, 833, 701, 408, 318, 337, 209, 775, 686, 635, 231, 139, 260, 722, 664, 313, 541, 91, 756, 519, 343, 823, 857, 791, 530, 986, 949, 750, 859, 39, 615, 915, 487, 191, 899, 998, 326, 129, 121, 330, 569, 44, 863, 516, 885, 53, 553, 736, 136, 605, 16, 99, 594, 762, 302, 952, 836, 758, 486, 103, 880, 644, 746], device=T_device)
    class_cache = selected_tensor[torch.randint(0, len(selected_tensor), (cache_size,), device=T_device)]

    c_emb_cache = torch.randn(cache_size, 1, 512).to(T_device)

    # 10%의 인덱스를 무작위로 선택하여 1000으로 설정
    num_to_replace = int(cache_size * 0.1)  # 전체 크기의 10%
    indices = torch.randperm(cache_size)[:num_to_replace]  # 랜덤으로 인덱스 선택
    class_cache[indices] = 1000
    
    with torch.no_grad():
        indices = []
        
        for i in range(1,int(args.T/2)):
            # 0부터 i*n까지의 값
            indices.extend(range(i * args.cache_n))
            
            # (1000-i)*n부터 500*n까지의 값
            indices.extend(range((1000 - i) * args.cache_n-1, 500 * args.cache_n-1, -1))
            
        for i in range(int(args.T/2)):
            indices.extend(range(500 * args.cache_n))
        
        for batch_start in trange(0, cache_size, args.caching_batch_size, desc="Pre-class_caching"):
            batch_end = min(batch_start + args.caching_batch_size, cache_size)  # 인덱스 범위를 벗어나지 않도록 처리
            class_batch = class_cache[batch_start:batch_end]
            
            c = T_model.get_learned_conditioning(
                {T_model.cond_stage_key: class_batch}
            )
            
            c_emb_cache[batch_start:batch_end] = c
            
            
        # Batch size만큼의 인덱스를 뽑아오는 과정
        for batch_start in trange(0, len(indices), args.caching_batch_size, desc="Pre-caching"):
            batch_end = min(batch_start + args.caching_batch_size, len(indices))  # 인덱스 범위를 벗어나지 않도록 처리
            batch_indices = indices[batch_start:batch_end]  # Batch size만큼 인덱스 선택

            # 인덱스를 이용해 배치 선택
            img_batch = img_cache[batch_indices]
            t_batch = t_cache[batch_indices]
            c = c_emb_cache[batch_indices]
            

            x_prev, pred_x0,_ = T_sampler.cache_step(img_batch, c, t_batch, t_batch,
                                                    use_original_steps=True,
                                                    unconditional_guidance_scale=args.cfg_scale)

            # 결과를 저장
            img_cache[batch_indices] = x_prev
            t_cache[batch_indices] -= 1

            if batch_start % 100 == 0:  # 예를 들어, 100 스텝마다 시각화
                visualize_t_cache_distribution(t_cache, args.cache_n)
                
        visualize_t_cache_distribution(t_cache, args.cache_n)
        
        save_dir = f"./{args.cachedir}/{args.cache_n}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save img_cache, t_cache, and class_cache as .pt files
        torch.save(img_cache, f"{save_dir}/img_cache_{args.cache_n}_{args.seed}.pt")
        torch.save(t_cache, f"{save_dir}/t_cache_{args.cache_n}_{args.seed}.pt")
        torch.save(class_cache, f"{save_dir}/class_cache_{args.cache_n}_{args.seed}.pt")
        torch.save(c_emb_cache, f"{save_dir}/c_emb_cache_{args.cache_n}_{args.seed}.pt")
        
        slice1 = img_cache[0:4]  # 첫 번째 슬라이스: 0부터 args.cache_n
        slice2 = img_cache[args.cache_n*200:args.cache_n*200+4]  # 두 번째 슬라이스: args.cache_n*200부터 args.cache_n*201
        slice3 = img_cache[args.cache_n*400:args.cache_n*400+4]  # 세 번째 슬라이스: args.cache_n*400부터 args.cache_n*401
        slice4 = img_cache[args.cache_n*600:args.cache_n*600+4]  # 네 번째 슬라이스: args.cache_n*600부터 args.cache_n*601

        # 슬라이스들을 합치기
        img_to_save = torch.cat((slice1, slice2, slice3, slice4), dim=0)
        img = T_model.decode_first_stage(img_to_save)        
        grid_T = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        grid_T = make_grid(grid_T, nrow=4)
        # 각각의 그리드를 이미지로 변환
        grid_T = 255. * rearrange(grid_T, 'c h w -> h w c').cpu().numpy()
        # 이미지로 저장 (T_model과 S_model의 결과)
        output_image_T = Image.fromarray(grid_T.astype(np.uint8))
        output_image_T_path = 'caching_image.png'
        output_image_T.save(output_image_T_path)

        # 저장 경로 설정
        save_dir = f"./{args.cachedir}/{args.cache_n}/images"
        os.makedirs(save_dir, exist_ok=True)


    print(f"Pre-caching completed and saved to {args.cachedir}")
    
def distillation(rank, world_size, args):

    # #gpu_monitor = GPUMonitor(monitoring_interval=2)
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # Initialize WandB
    if rank == 0:  # Only initialize WandB in one process (e.g., rank 0)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            config={
                "learning_rate": args.lr,
                "architecture": "UNet",
                "dataset": "ldm caching",
                "steps": int(args.total_steps//args.gradient_accumulation_steps),
            }
        )
    

    # DDP 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    T_model = get_model_teacher()
    S_model = get_model_student()  #load with ckpt
    print('load model to CPU')
    T_model = T_model.to(device)
    print('load T_model to device')
    S_model = S_model.to(device)
    print('load S_model to device')
    T_model.eval()
    
    initialize_params(S_model)  #initialize unet parameters
    print('initialize S_model')
    
    trainable_params_student = list(filter(lambda p: p.requires_grad, S_model.parameters()))
    print('trainable params S_model')
    
    T_sampler = DDIMSampler(T_model)
    S_sampler = DDIMSampler(S_model)
    print('sampler')
    T_sampler.make_schedule(ddim_num_steps = args.DDIM_num_steps, ddim_eta= 1, verbose=False)
    S_sampler.make_schedule(ddim_num_steps = args.DDIM_num_steps, ddim_eta= 1, verbose=False)
    print('sampler make schedule')
    trainer = distillation_DDPM_trainer(T_model, S_model, T_sampler, S_sampler, args.distill_features)
    print('trainer')
    
    S_model = DDP(S_model, device_ids=[rank])
    S_model.train()
    print('load S_model to DDP')
    
    optimizer = torch.optim.AdamW(
        trainable_params_student,
        lr=args.lr
        # betas=(args.adam_beta1, args.adam_beta2),
        # weight_decay=args.adam_weight_decay,
        # eps=args.adam_epsilon,
    )
    print('set optimizer')
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.total_steps
    # )
    # scheduler = LambdaLR(optimizer, lr_lambda=scheduler.schedule)


    img_cache, t_cache, c_emb_cache, class_cache = load_cache(args.cachedir) # 함수 추가
    print('load_cache, size:', img_cache.shape[0])
    
    if img_cache.numel() == 0 or t_cache.numel() == 0 or c_emb_cache.numel() == 0 or class_cache.numel() == 0:
        print("The cache is empty. You need to generate the cache.")        
        dist.barrier()  # Synchronize before exit
        dist.destroy_process_group()
        sys.exit(1)
    
    cache_dataset = Cache_Dataset(img_cache, t_cache, c_emb_cache, class_cache) # 함수 추가
    
    # DDP를 위한 샘플러와 DataLoader 생성
    sampler = DistributedSampler(cache_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(cache_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, sampler=sampler)
    dataloader_cycle = cycle(dataloader)
    
    with trange(args.total_steps*args.gradient_accumulation_steps, dynamic_ncols=True, disable=(rank != 0)) as pbar:
        for step in pbar:
            # step이 epoch의 시작을 나타낼 때마다 sampler의 epoch을 업데이트
            if step % len(dataloader) == 0:
                epoch = step // len(dataloader)
                sampler.set_epoch(epoch)  # Sampler의 epoch을 업데이트하여 새로운 셔플링 수행

            # optimizer.zero_grad()
            
            # 다음 배치를 가져옴
            x_t, t, c, _, indices = next(dataloader_cycle)

            x_t = x_t.to(device)
            t = t.to(device)
            c = c.to(device)
            
            # Calculate distillation loss
            output_loss, total_loss, x_prev = trainer(x_t, c, t, args.cfg_scale, args.loss_weight)
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # optimizer.step()
            # lr_scheduler.step()

            cache_dataset.update_data(indices, x_prev)
            
            
            if rank == 0:
                
                # if step%1000 == 0:
                #     visualize_t_cache_distribution(t_cache, args.cache_n)
                if step%args.gradient_accumulation_steps==0 :
                    # Logging with WandB
                    wandb.log({
                        'distill_loss': total_loss.item() * args.gradient_accumulation_steps,
                        'output_loss': output_loss.item()
                            }, step=int(step//args.gradient_accumulation_steps))
                    pbar.set_postfix(distill_loss='%.3f' % (total_loss.item()* args.gradient_accumulation_steps))
                
                ################### Sample and save student outputs############################
                if step>0 and args.sample_step > 0 and step/args.gradient_accumulation_steps % args.sample_step == 0:
                    S_model.eval()
                    sample_save_images(args.num_sample_class, args.n_sample_per_class, 
                                    args.sample_save_ddim_steps, args.DDPM_sampling, args.ddim_eta, args.cfg_scale, 
                                    T_model, S_model.module, T_sampler, S_sampler, int(step/args.gradient_accumulation_steps))
                    S_model.train()
            
                ################### Save student model ################################
                if step>0 and args.save_step > 0 and step/args.gradient_accumulation_steps % args.save_step == 0:
                    save_checkpoint(S_model.module, optimizer, int(step//args.gradient_accumulation_steps), args.logdir)
                    
                ################### Evaluate student model ##############################
                if step>0 and args.eval_step > 0 and step/args.gradient_accumulation_steps % args.eval_step == 0:# and step != 0:
                    
                    specific_classes = [862, 43, 335, 146, 494, 491, 587, 588, 187, 961, 78, 205, 297, 214, 163, 788, 980, 507, 916, 112, 512, 589, 771, 27, 269, 386, 336, 280, 362, 510, 850, 661, 731, 613, 945, 704, 86, 160, 372, 910, 159, 493, 623, 73, 128, 234, 717, 710, 887, 423, 546, 148, 558, 358, 463, 224, 987, 960, 444, 965, 363, 854, 492, 87, 672, 870, 217, 292, 303, 508, 188, 296, 642, 349, 154, 690, 298, 670, 964, 341, 873, 236, 35, 28, 890, 698, 902, 457, 621, 629, 371, 114, 610, 186, 718, 815, 944, 832, 869, 919, 441, 394, 625, 993, 401, 650, 55, 825, 272, 233, 738, 483, 473, 8, 220, 547, 684, 533, 132, 646, 455, 895, 52, 400, 593, 943, 848, 380, 175, 951, 195, 404, 856, 464, 123, 10, 433, 283, 366, 122, 307, 460, 616, 585, 407, 785, 835, 712, 912, 397, 440, 901, 600, 732, 140, 499, 864, 653, 584, 844, 874, 420, 147, 574, 24, 183, 243, 379, 338, 699, 94, 79, 254, 458, 430, 350, 388, 711, 639, 415, 299, 412, 743, 340, 967, 17, 992, 480, 858, 393, 918, 193, 334, 324, 575, 130, 950, 759, 820, 244, 652, 171, 18, 576, 15, 581, 93, 290, 847, 505, 922, 883, 470, 293, 777, 696, 215, 322, 291, 540, 416, 40, 956, 488, 780, 184, 453, 792, 127, 200, 602, 378, 344, 273, 255, 935, 763, 714, 529, 700, 226, 76, 502, 566, 165, 106, 867, 811, 376, 802, 678, 267, 276, 767, 881, 248, 26, 567, 995, 143, 709, 124, 927, 431, 270, 29, 966, 926, 168, 769, 149, 786, 761, 14, 22, 474, 981, 257, 676, 662, 96, 872, 679, 177, 413, 928, 314, 185, 120, 687, 395, 599, 346, 737, 352, 638, 157, 716, 974, 783, 467, 697, 559, 181, 797, 111, 144, 389, 834, 715, 894, 70, 206, 666, 0, 190, 520, 142, 259, 429, 948, 729, 841, 830, 764, 232, 150, 446, 80, 782, 225, 391, 477, 720, 295, 319, 803, 182, 989, 831, 800, 166, 506, 563, 721, 135, 305, 904, 145, 427, 72, 178, 947, 975, 33, 706, 997, 60, 828, 829, 45, 432, 482, 98, 392, 846, 968, 381, 577, 57, 240, 179, 484, 167, 282, 969, 542, 768, 930, 65, 239, 359, 107, 619, 218, 824, 503, 733, 515, 958, 469, 288, 606, 439, 622, 618, 419, 971, 294, 263, 504, 247, 744, 651, 310, 806, 339, 434, 633, 204, 659, 702, 351, 85, 81, 673, 449, 591, 537, 572, 668, 227, 580, 655, 962, 724, 937, 766, 742, 194, 285, 435, 897, 462, 708, 776, 693, 192, 582, 843, 597, 437, 513, 357, 365, 398, 713, 990, 523, 946, 837, 840, 564, 608, 855, 522, 719, 849, 603, 853, 691, 550, 37, 809, 778, 89, 321, 548, 309, 102, 41, 745, 399, 631, 7, 812, 421, 554, 119, 472, 438, 32, 481, 685, 817, 490, 723, 12, 570, 9, 568, 387, 164, 211, 6, 46, 448, 695, 242, 521, 978, 814, 875, 607, 634, 931, 884, 614, 320, 251, 77, 237, 118, 810, 617, 61, 311, 703, 963, 772, 972, 878, 571, 794, 868, 67, 774, 674, 976, 955, 49, 842, 117, 216, 932, 632, 134, 109, 994, 308, 747, 245, 517, 991, 648, 249, 643, 628, 590, 90, 30, 279, 345, 770, 544, 795, 705, 126, 913, 936, 636, 985, 219, 497, 751, 383, 410, 20, 63, 424, 138, 230, 261, 235, 649, 13, 1, 929, 228, 906, 38, 560, 598, 436, 798, 375, 921, 396, 5, 354, 640, 83, 3, 624, 511, 725, 630, 826, 333, 425, 361, 411, 626, 773, 471, 556, 728, 781, 161, 278, 790, 601, 450, 384, 996, 317, 565, 489, 804, 755, 641, 4, 277, 405, 539, 819, 115, 892, 113, 551, 734, 527, 924, 325, 451, 957, 367, 342, 323, 973, 289, 356, 327, 23, 545, 941, 36, 534, 784, 11, 977, 671, 637, 536, 905, 821, 669, 369, 101, 748, 907, 370, 212, 108, 579, 920, 595, 377, 31, 390, 409, 137, 189, 2, 222, 983, 667, 557, 385, 201, 970, 586, 692, 287, 866, 796, 58, 238, 726, 984, 445, 258, 75, 92, 355, 665, 153, 300, 162, 675, 596, 208, 903, 500, 466, 442, 286, 838, 328, 54, 531, 34, 456, 877, 689, 97, 552, 891, 760, 543, 199, 418, 660, 459, 100, 19, 514, 274, 246, 681, 647, 253, 805, 645, 900, 765, 938, 752, 50, 663, 151, 683, 526, 461, 741, 917, 152, 88, 301, 68, 125, 203, 498, 275, 822, 934, 654, 56, 155, 110, 735, 131, 74, 156, 262, 443, 207, 871, 525, 818, 306, 562, 173, 454, 485, 739, 382, 677, 364, 501, 942, 402, 749, 914, 172, 813, 176, 865, 573, 753, 592, 827, 62, 48, 347, 284, 852, 82, 730, 816, 71, 518, 909, 213, 953, 21, 688, 202, 360, 882, 141, 69, 105, 898, 104, 611, 315, 403, 999, 561, 374, 694, 876, 252, 496, 754, 158, 84, 808, 982, 532, 281, 42, 264, 348, 896, 174, 373, 879, 845, 911, 406, 198, 422, 583, 59, 535, 680, 627, 923, 316, 51, 265, 620, 417, 549, 353, 727, 304, 495, 250, 475, 241, 538, 312, 682, 66, 95, 658, 426, 266, 479, 578, 889, 368, 476, 988, 801, 740, 886, 807, 787, 414, 197, 940, 210, 779, 452, 833, 701, 408, 318, 337, 209, 775, 686, 635, 231, 139, 260, 722, 664, 313, 541, 91, 756, 519, 343, 823, 857, 791, 530, 986, 949, 750, 859, 39, 615, 915, 487, 191, 899, 998, 326, 129, 121, 330, 569, 44, 863, 516, 885, 53, 553, 736, 136, 605, 16, 99, 594, 762, 302, 952, 836, 758, 486, 103, 880, 644, 746]
                    
                    S_model.eval()
                    
                    fid_a, fid_b, fid_c = sample_and_cal_fid(model=S_model.module , device=device, num_images=args.num_images, ddim_eta = args.ddim_eta, cfg_scale = args.cfg_scale, DDIM_num_steps=args.DDIM_num_steps, specific_classes=specific_classes)
                    
                    S_model.train()
                    
                    metrics = {
                        'Student_FID_A': fid_a,
                        'Student_FID_B': fid_b,
                        'Student_FID_C': fid_c,
                    }
                    
                    print(metrics)
                    
                    # Log metrics to wandb
                    wandb.log(metrics, step=int(step//args.gradient_accumulation_steps))
    # DDP 정리 및 WandB 로깅 종료
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        wandb.finish()



def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    parser = get_parser()
    distill_args = parser.parse_args(argv[1:])  # argv[1:]로 수정하여 인자 전달
    seed_everything(distill_args.seed)
    
    if distill_args.pre_caching:
        pre_caching(distill_args)
        
    else:
        # world_size 설정
        os.environ['MASTER_ADDR'] = distill_args.MASTER_ADDR
        os.environ['MASTER_PORT'] = distill_args.MASTER_PORT

        if distill_args.world_size is None:
            # 선택하지 않은 경우, torch.cuda.device_count()를 사용
            distill_args.world_size = torch.cuda.device_count()
            
        # world_size는 지정된 world_size와 device_count 중 더 작은 값으로 설정
        world_size = min(torch.cuda.device_count(), distill_args.world_size)
        print('world_size(gpu num): ', world_size)
        
        # Ensure we have multiple GPUs available
        if world_size < 1:
            print("No GPUs available for DDP. Exiting...")
            sys.exit(1)

        # Spawn processes for DDP
        mp.spawn(
            distillation,
            args=(world_size, distill_args),
            nprocs=world_size,
            join=True
        )

if __name__ == '__main__':
    main(sys.argv)  # main()을 직접 호출
