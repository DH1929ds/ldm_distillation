import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_fid import fid_score
from fid_score_gpu import calculate_fid_given_paths
from collections import defaultdict
import warnings, os, shutil, time, subprocess, re, random, argparse



def copy_files_from_folders(source_folder_1, source_folder_2, destination_folder, num_files_per_class=1):
    """
    Parameters:
    - source_folder_1 (str): 첫 번째 원본 폴더 경로.
    - source_folder_2 (str): 두 번째 원본 폴더 경로.
    - destination_folder (str): 복사할 파일들이 저장될 대상 폴더 경로.
    - num_files_per_class (int): 각 클래스당 복사할 파일 개수 (기본값: 1).
    """
    
    # 대상 폴더가 없다면 생성
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 정규 표현식을 통해 파일명을 분석 (class_x_sample_y.png 형식)
    pattern = re.compile(r'class_(\d+)_sample_\d+\.png')

    # 각 폴더에서 클래스별 파일을 담을 딕셔너리 초기화
    class_files_1 = defaultdict(list)
    class_files_2 = defaultdict(list)

    # 첫 번째 폴더에서 파일을 탐색
    for filename in os.listdir(source_folder_1):
        match = pattern.match(filename)
        if match:
            class_num = int(match.group(1))  # 클래스 번호 추출
            class_files_1[class_num].append(filename)

    # 두 번째 폴더에서 파일을 탐색
    for filename in os.listdir(source_folder_2):
        match = pattern.match(filename)
        if match:
            class_num = int(match.group(1))  # 클래스 번호 추출
            class_files_2[class_num].append(filename)

    # 클래스별로 지정된 파일 개수를 복사
    for class_num in set(class_files_1.keys()).union(set(class_files_2.keys())):
        # 첫 번째 폴더에서 지정된 파일 개수 복사
        if class_num in class_files_1:
            for file in class_files_1[class_num][:num_files_per_class]:
                src_file = os.path.join(source_folder_1, file)
                dst_file = os.path.join(destination_folder, file)
                shutil.copy(src_file, dst_file)
                print(f'Copied {file} from {source_folder_1} to {destination_folder}')
        
        # 두 번째 폴더에서 지정된 파일 개수 복사
        if class_num in class_files_2:
            for file in class_files_2[class_num][:num_files_per_class]:
                src_file = os.path.join(source_folder_2, file)
                dst_file = os.path.join(destination_folder, file)
                shutil.copy(src_file, dst_file)
                print(f'Copied {file} from {source_folder_2} to {destination_folder}')


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    
    # sd = pl_sd["state_dict"]
    # student 모델이라 state_dict -> student_model
    # teacher의 경우 sd = pl_sd["state_dict"] 사용
    sd = pl_sd['student_model']
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(model=None):
    config = OmegaConf.load("./cin256-v2.yaml")

    # teacher
    if model is None:
        model = load_model_from_config(config, "./model.ckpt")

    # student
    # /home/jovyan/fileviewer/ldm_distillation/logs/cin256-v2/student_ckpt_step_50000.pt

    
    model = load_model_from_config(config, f"{model}")
    return model


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def save_samples_as_images(samples, folder_path, class_label, start_idx):
    '''
    class_folder = os.path.join(folder_path, f'{class_label}')
    
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for i, sample in enumerate(samples):
        img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
        output_image = Image.fromarray(img.astype(np.uint8))
        # 배치 시작 인덱스와 배치 내 인덱스를 결합하여 파일 이름 생성
        output_image.save(os.path.join(folder_path, f'class_{class_label}_sample_{start_idx + i}.png'))


def copy_files_by_class_range(val_directory, start_class, end_class, destination):
    for class_index in range(start_class, end_class + 1):
        class_folder = os.path.join(val_directory, str(class_index))
        if os.path.exists(class_folder):
            copy_files(class_folder, destination)
        else:
            print(f"Class folder {class_folder} does not exist, skipping.")


def sampling(model=None, output_folder = "output_samples", device="cuda", large_batch_size=250, small_batch_size=50, num_images=10000,cfg_scale=1.0, ddim_eta=1.0, DDIM_num_steps=25, classes = list(range(1000))):
    
    if model is None:
        model = get_model()
    model.to(device)
    sampler = DDIMSampler(model)
    
    ## classes = list(range(1000))  # 샘플링할 클래스
    
    n_samples_per_class = num_images // len(classes)  # 클래스당 샘플 수

    ### remaining image 추가 ###
    remaining_images = num_images % len(classes)

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

            # 나머지 이미지 분배
            if remaining_images > 0:
                extra_classes = random.sample(classes, remaining_images)  # 무작위로 나머지 클래스 선택
                all_classes += extra_classes

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


def copy_files_by_class_list(val_directory, class_list, destination):
    for class_index in class_list:
        class_folder = os.path.join(val_directory, str(class_index))
        if os.path.exists(class_folder):
            copy_files(class_folder, destination)
        else:
            print(f"Class folder {class_folder} does not exist, skipping.")


def copy_files_by_class_exclude(val_directory, specific_class_list, destination):
    all_classes = list(range(1000))  # 클래스 0부터 999까지
    exclude_class_list = [class_idx for class_idx in all_classes if class_idx not in specific_class_list]
    copy_files_by_class_list(val_directory, exclude_class_list, destination)


def copy_files_to_unspecific_destinations(val_directory, specific_classes, destination_specific, destination_exclude, destination_all):

    # Step 2: Copy all classes excluding specific ones to the exclude destination
    copy_files_by_class_exclude(val_directory, specific_classes, destination_exclude)

    print("File copying completed.")


def sample_and_cal_fid(device, num_images=50000, model=None, output_dir="./output_samples", ddim_eta=1.0, cfg_scale=1.0, DDIM_num_steps=25, specific_classes=None):
    
    start_time = time.time()
   
    print(len(specific_classes))

    
    ### specific ### (seen)
    output_specific_dir = f"{output_dir}/output_samples_specific_{num_images}"
    
    sampling(output_folder=output_specific_dir, model=model, device=device, num_images=num_images, ddim_eta=ddim_eta, cfg_scale=cfg_scale, DDIM_num_steps=DDIM_num_steps, classes = specific_classes)


    ### exclude ### (unseen)
    exclude_list = [x for x in range(0,999) if x not in specific_classes]
    output_exclude_dir = f"{output_dir}/output_samples_exclude_{num_images}"
    
    exclude_classes = exclude_list
    sampling(output_folder=output_exclude_dir, model=model, device=device, num_images=num_images, ddim_eta=ddim_eta, cfg_scale=cfg_scale, DDIM_num_steps=DDIM_num_steps, classes = exclude_classes)


    ### all ###
    output_all_dir = f"{output_dir}/output_samples_all_{num_images}"
    all_classes = list(range(0,1000))
    '''
    Not sampling, but copy and paste.
    
    sampling(output_folder=output_all_dir, model=model, device=device, num_images=num_images, ddim_eta=ddim_eta, cfg_scale=cfg_scale, DDIM_num_steps=DDIM_num_steps, classes = all_classes)
    '''
    num_files_per_class = num_images // len(all_classes)
    copy_files_from_folders(source_folder_1=output_specific_dir, source_folder_2=output_exclude_dir, destination_folder=output_all_dir, num_files_per_class=num_files_per_class)

    
    '''
    calculate fid
    - (specific, specific')
    - (exclude, exclude')
    - (all, all')
    '''
    
    # FID 계산
    fid_specific_start_time = time.time()
    print("start fid_specific_pair")
    fid_value_specific = calculate_fid_given_paths([f"{num_images}_npz_files/trainset_seen_{num_images}.npz", output_specific_dir], 
                                                   batch_size=50, 
                                                   device=device, 
                                                   dims=2048)
    end_time_specific = time.time()
    execution_time_specific = end_time_specific - fid_specific_start_time
    print(f"finish calculate fid specific pair, {execution_time_specific}s")
    
    
    fid_exclude_start_time = time.time()
    print("start fid_exclude_pair")
    fid_value_exclude = calculate_fid_given_paths([f"{num_images}_npz_files/trainset_unseen_{num_images}.npz", output_exclude_dir], 
                                                  batch_size=50, 
                                                  device=device, 
                                                  dims=2048)
    end_time_exclude = time.time()
    execution_time_exclude = end_time_exclude - fid_exclude_start_time
    print(f"finish calculate fid exclude pair, {execution_time_exclude}s")

    
    fid_all_start_time = time.time()
    print("start fid_all_pair")
    fid_value_all = calculate_fid_given_paths([f"{num_images}_npz_files/trainset_all_{num_images}.npz", output_all_dir], 
                                              batch_size=50, 
                                              device=device, 
                                              dims=2048)
    end_time_all = time.time()
    execution_time_all = end_time_all - fid_all_start_time
    print(f"finish calculate fid all pair, {execution_time_all}s")

    end_time = time.time()
    execution_time = end_time - start_time

    print("FID scores")
    print(f"FID score_specific_pair: {fid_value_specific}")
    print(f"FID score_exclude_pair: {fid_value_exclude}")
    print(f"FID score_all_pair: {fid_value_all}")

    print(f"FID 실행 시간(sampling+cal_fid): {execution_time} 초")

    # Delete samples
    #delete_folder(destination_all)
    #delete_folder(destination_specific)
    #delete_folder(destination_exclude)
    
    return fid_value_specific, fid_value_exclude, fid_value_all



def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation arguments")

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output_samples", 
        help="Directory to save sampled images"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=50000, 
        help="Total number of samples"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to use for computation"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/ldm/cin256-v2/model.ckpt", 
        help="Path of Unet Model (default is teacher model)"
    )
    parser.add_argument(
        "--cfg_scale", 
        type=float, 
        default=1.0, 
        help="Unconditional guidance scale"
    )
    parser.add_argument(
        "--ddim_eta", 
        type=float, 
        default=1.0, 
        help="DDIM eta"
    )
    parser.add_argument(
        "--DDIM_num_steps", 
        type=int, 
        default=25, 
        help="Number of steps for DDIM sampling"
    )

    return parser


def main():
    # test
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #output_dir = "./output_samples/"
    # ddim_eta=1.0, cfg_scale=1.0
    
    args = get_parser().parse_args()
    
    specific_classes  = [862, 43, 335, 146, 494, 491, 587, 588, 187, 961, 78, 205, 297, 214, 163, 788, 980, 507, 916, 112, 512, 589, 771, 27, 269, 386, 336, 280, 362, 510, 850, 661, 731, 613, 945, 704, 86, 160, 372, 910, 159, 493, 623, 73, 128, 234, 717, 710, 887, 423, 546, 148, 558, 358, 463, 224, 987, 960, 444, 965, 363, 854, 492, 87, 672, 870, 217, 292, 303, 508, 188, 296, 642, 349, 154, 690, 298, 670, 964, 341, 873, 236, 35, 28, 890, 698, 902, 457, 621, 629, 371, 114, 610, 186, 718, 815, 944, 832, 869, 919, 441, 394, 625, 993, 401, 650, 55, 825, 272, 233, 738, 483, 473, 8, 220, 547, 684, 533, 132, 646, 455, 895, 52, 400, 593, 943, 848, 380, 175, 951, 195, 404, 856, 464, 123, 10, 433, 283, 366, 122, 307, 460, 616, 585, 407, 785, 835, 712, 912, 397, 440, 901, 600, 732, 140, 499, 864, 653, 584, 844, 874, 420, 147, 574, 24, 183, 243, 379, 338, 699, 94, 79, 254, 458, 430, 350, 388, 711, 639, 415, 299, 412, 743, 340, 967, 17, 992, 480, 858, 393, 918, 193, 334, 324, 575, 130, 950, 759, 820, 244, 652, 171, 18, 576, 15, 581, 93, 290, 847, 505, 922, 883, 470, 293, 777, 696, 215, 322, 291, 540, 416, 40, 956, 488, 780, 184, 453, 792, 127, 200, 602, 378, 344, 273, 255, 935, 763, 714, 529, 700, 226, 76, 502, 566, 165, 106, 867, 811, 376, 802, 678, 267, 276, 767, 881, 248, 26, 567, 995, 143, 709, 124, 927, 431, 270, 29, 966, 926, 168, 769, 149, 786, 761, 14, 22, 474, 981, 257, 676, 662, 96, 872, 679, 177, 413, 928, 314, 185, 120, 687, 395, 599, 346, 737, 352, 638, 157, 716, 974, 783, 467, 697, 559, 181, 797, 111, 144, 389, 834, 715, 894, 70, 206, 666, 0, 190, 520, 142, 259, 429, 948, 729, 841, 830, 764, 232, 150, 446, 80, 782, 225, 391, 477, 720, 295, 319, 803, 182, 989, 831, 800, 166, 506, 563, 721, 135, 305, 904, 145, 427, 72, 178, 947, 975, 33, 706, 997, 60, 828, 829, 45, 432, 482, 98, 392, 846, 968, 381, 577, 57, 240, 179, 484, 167, 282, 969, 542, 768, 930, 65, 239, 359, 107, 619, 218, 824, 503, 733, 515, 958, 469, 288, 606, 439, 622, 618, 419, 971, 294, 263, 504, 247, 744, 651, 310, 806, 339, 434, 633, 204, 659, 702, 351, 85, 81, 673, 449, 591, 537, 572, 668, 227, 580, 655, 962, 724, 937, 766, 742, 194, 285, 435, 897, 462, 708, 776, 693, 192, 582, 843, 597, 437, 513, 357, 365, 398, 713, 990, 523, 946, 837, 840, 564, 608, 855, 522, 719, 849, 603, 853, 691, 550, 37, 809, 778, 89, 321, 548, 309, 102, 41, 745, 399, 631, 7, 812, 421, 554, 119, 472, 438, 32, 481, 685, 817, 490, 723, 12, 570, 9, 568, 387, 164, 211, 6, 46, 448, 695, 242, 521, 978, 814, 875, 607, 634, 931, 884, 614, 320, 251, 77, 237, 118, 810, 617, 61, 311, 703, 963, 772, 972, 878, 571, 794, 868, 67, 774, 674, 976, 955, 49, 842, 117, 216, 932, 632, 134, 109, 994, 308, 747, 245, 517, 991, 648, 249, 643, 628, 590, 90, 30, 279, 345, 770, 544, 795, 705, 126, 913, 936, 636, 985, 219, 497, 751, 383, 410, 20, 63, 424, 138, 230, 261, 235, 649, 13, 1, 929, 228, 906, 38, 560, 598, 436, 798, 375, 921, 396, 5, 354, 640, 83, 3, 624, 511, 725, 630, 826, 333, 425, 361, 411, 626, 773, 471, 556, 728, 781, 161, 278, 790, 601, 450, 384, 996, 317, 565, 489, 804, 755, 641, 4, 277, 405, 539, 819, 115, 892, 113, 551, 734, 527, 924, 325, 451, 957, 367, 342, 323, 973, 289, 356, 327, 23, 545, 941, 36, 534, 784, 11, 977, 671, 637, 536, 905, 821, 669, 369, 101, 748, 907, 370, 212, 108, 579, 920, 595, 377, 31, 390, 409, 137, 189, 2, 222, 983, 667, 557, 385, 201, 970, 586, 692, 287, 866, 796, 58, 238, 726, 984, 445, 258, 75, 92, 355, 665, 153, 300, 162, 675, 596, 208, 903, 500, 466, 442, 286, 838, 328, 54, 531, 34, 456, 877, 689, 97, 552, 891, 760, 543, 199, 418, 660, 459, 100, 19, 514, 274, 246, 681, 647, 253, 805, 645, 900, 765, 938, 752, 50, 663, 151, 683, 526, 461, 741, 917, 152, 88, 301, 68, 125, 203, 498, 275, 822, 934, 654, 56, 155, 110, 735, 131, 74, 156, 262, 443, 207, 871, 525, 818, 306, 562, 173, 454, 485, 739, 382, 677, 364, 501, 942, 402, 749, 914, 172, 813, 176, 865, 573, 753, 592, 827, 62, 48, 347, 284, 852, 82, 730, 816, 71, 518, 909, 213, 953, 21, 688, 202, 360, 882, 141, 69, 105, 898, 104, 611, 315, 403, 999, 561, 374, 694, 876, 252, 496, 754, 158, 84, 808, 982, 532, 281, 42, 264, 348, 896, 174, 373, 879, 845, 911, 406, 198, 422, 583, 59, 535, 680, 627, 923, 316, 51, 265, 620, 417, 549, 353, 727, 304, 495, 250, 475, 241, 538, 312, 682, 66, 95, 658, 426, 266, 479, 578, 889, 368, 476, 988, 801, 740, 886, 807, 787, 414, 197, 940, 210, 779, 452, 833, 701, 408, 318, 337, 209, 775, 686, 635, 231, 139, 260, 722, 664, 313, 541, 91, 756, 519, 343, 823, 857, 791, 530, 986, 949, 750, 859, 39, 615, 915, 487, 191, 899, 998, 326, 129, 121, 330, 569, 44, 863, 516, 885, 53, 553, 736, 136, 605, 16, 99, 594, 762, 302, 952, 836, 758, 486, 103, 880, 644, 746]

    model = get_model(args.model_path)

    sample_and_cal_fid(model = model, device=args.device, output_dir=args.output_dir, num_images=args.num_images, cfg_scale=args.cfg_scale, DDIM_num_steps=args.DDIM_num_steps, ddim_eta=args.ddim_eta, specific_classes=specific_classes)


if __name__ == "__main__":
    main()
