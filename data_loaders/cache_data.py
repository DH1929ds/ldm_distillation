import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import os, sys
import torch

def load_cache(cachedir):
    # 각 캐시 텐서를 저장할 리스트 초기화
    img_cache_list = []
    t_cache_list = []
    c_emb_cache_list = []
    class_cache_list = []

    # 캐시 파일을 찾기 위한 기본 디렉토리 설정
    base_dir = f"./{cachedir}"

    # cachedir 안의 각 cache_n 디렉토리를 순회
    for cache_n in sorted(os.listdir(base_dir)):
        cache_n_dir = os.path.join(base_dir, cache_n)

        # cache_n 디렉토리가 존재하는지 확인
        if os.path.isdir(cache_n_dir):
            # cache_n 디렉토리 안의 파일을 순회
            for file_name in sorted(os.listdir(cache_n_dir)):
                # 파일 이름이 img_cache 패턴에 맞는지 확인
                if file_name.startswith(f"img_cache_{cache_n}_") and file_name.endswith('.pt'):
                    seed = file_name.split('_')[-1].replace('.pt', '')

                    # 각 캐시 파일 불러오기
                    img_cache = torch.load(os.path.join(cache_n_dir, f"img_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                    t_cache = torch.load(os.path.join(cache_n_dir, f"t_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                    c_emb_cache = torch.load(os.path.join(cache_n_dir, f"c_emb_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                    class_cache = torch.load(os.path.join(cache_n_dir, f"class_cache_{cache_n}_{seed}.pt"), map_location='cpu')

                    # 불러온 캐시를 리스트에 추가
                    img_cache_list.append(img_cache)
                    t_cache_list.append(t_cache)
                    c_emb_cache_list.append(c_emb_cache)
                    class_cache_list.append(class_cache)

    # 0번째 차원에 대해 캐시들을 이어 붙임
    img_cache = torch.cat(img_cache_list, dim=0)
    t_cache = torch.cat(t_cache_list, dim=0)
    c_emb_cache = torch.cat(c_emb_cache_list, dim=0)
    class_cache = torch.cat(class_cache_list, dim=0)

    # 이어 붙인 캐시들을 반환
    return img_cache, t_cache, c_emb_cache, class_cache

    
class Cache_Dataset(Dataset):
    def __init__(self, cachedir, rank, world_size):
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        if rank == 0:
            img_cache, t_cache, c_emb_cache, class_cache = self.load_cache(cachedir)
            print('load_cache, size:', img_cache.shape[0])

            if img_cache.numel() == 0 or t_cache.numel() == 0 or c_emb_cache.numel() == 0 or class_cache.numel() == 0:
                print("The cache is empty. You need to generate the cache.")
                dist.barrier()  # Synchronize before exit
                dist.destroy_process_group()
                sys.exit(1)

            # 데이터를 GPU로 이동
            img_cache_split = [x.to(device) for x in img_cache]
            t_cache_split = [x.to(device) for x in t_cache]
            c_emb_cache_split = [x.to(device) for x in c_emb_cache]
            class_cache_split = [x.to(device) for x in class_cache]

        else:
            img_cache_split = None
            t_cache_split = None
            c_emb_cache_split = None
            
            class_cache_split = None

        # GPU에서 scatter 작업 수행
        img_cache, t_cache, c_emb_cache, class_cache = self.scatter_cache_data(
            img_cache_split, t_cache_split, c_emb_cache_split, class_cache_split, rank, device
        )

        # 다시 CPU로 이동
        img_cache = img_cache.cpu()
        t_cache = t_cache.cpu()
        c_emb_cache = c_emb_cache.cpu()
        class_cache = class_cache.cpu()

        print(f"Rank {rank} received cache, size: {img_cache.shape[0]}")

        self.img_cache = img_cache
        self.t_cache = t_cache
        self.c_emb_cache = c_emb_cache
        self.class_cache = class_cache

    @staticmethod
    def load_cache(cachedir):
        # 각 캐시 텐서를 저장할 리스트 초기화
        img_cache_list = []
        t_cache_list = []
        c_emb_cache_list = []
        class_cache_list = []

        # 캐시 파일을 찾기 위한 기본 디렉토리 설정
        base_dir = f"./{cachedir}"

        # cachedir 안의 각 cache_n 디렉토리를 순회
        for cache_n in sorted(os.listdir(base_dir)):
            cache_n_dir = os.path.join(base_dir, cache_n)

            # cache_n 디렉토리가 존재하는지 확인
            if os.path.isdir(cache_n_dir):
                # cache_n 디렉토리 안의 파일을 순회
                for file_name in sorted(os.listdir(cache_n_dir)):
                    # 파일 이름이 img_cache 패턴에 맞는지 확인
                    if file_name.startswith(f"img_cache_{cache_n}_") and file_name.endswith('.pt'):
                        seed = file_name.split('_')[-1].replace('.pt', '')

                        # 각 캐시 파일 불러오기
                        img_cache = torch.load(os.path.join(cache_n_dir, f"img_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                        t_cache = torch.load(os.path.join(cache_n_dir, f"t_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                        c_emb_cache = torch.load(os.path.join(cache_n_dir, f"c_emb_cache_{cache_n}_{seed}.pt"), map_location='cpu')
                        class_cache = torch.load(os.path.join(cache_n_dir, f"class_cache_{cache_n}_{seed}.pt"), map_location='cpu')

                        # 불러온 캐시를 리스트에 추가
                        img_cache_list.append(img_cache)
                        t_cache_list.append(t_cache)
                        c_emb_cache_list.append(c_emb_cache)
                        class_cache_list.append(class_cache)

        # 0번째 차원에 대해 캐시들을 이어 붙임
        img_cache = torch.cat(img_cache_list, dim=0)
        t_cache = torch.cat(t_cache_list, dim=0)
        c_emb_cache = torch.cat(c_emb_cache_list, dim=0)
        class_cache = torch.cat(class_cache_list, dim=0)

        # 이어 붙인 캐시들을 반환
        return img_cache, t_cache, c_emb_cache, class_cache

    @staticmethod
    def split_cache_for_ranks(img_cache, t_cache, c_emb_cache, class_cache, world_size):
        # 각 rank에 분배할 크기 계산
        size_per_rank = img_cache.shape[0] // world_size
        remainder = img_cache.shape[0] % world_size

        # 각 rank로 분할할 텐서들 생성
        img_cache_split = [img_cache[i * size_per_rank:(i + 1) * size_per_rank] for i in range(world_size)]
        t_cache_split = [t_cache[i * size_per_rank:(i + 1) * size_per_rank] for i in range(world_size)]
        c_emb_cache_split = [c_emb_cache[i * size_per_rank:(i + 1) * size_per_rank] for i in range(world_size)]
        class_cache_split = [class_cache[i * size_per_rank:(i + 1) * size_per_rank] for i in range(world_size)]

        # 만약 나누어 떨어지지 않는 경우, 남은 데이터는 마지막 rank에 추가
        if remainder:
            img_cache_split[-1] = torch.cat([img_cache_split[-1], img_cache[-remainder:]], dim=0)
            t_cache_split[-1] = torch.cat([t_cache_split[-1], t_cache[-remainder:]], dim=0)
            c_emb_cache_split[-1] = torch.cat([c_emb_cache_split[-1], c_emb_cache[-remainder:]], dim=0)
            class_cache_split[-1] = torch.cat([class_cache_split[-1], class_cache[-remainder:]], dim=0)

        return img_cache_split, t_cache_split, c_emb_cache_split, class_cache_split

    @staticmethod
    def scatter_cache_data(img_cache_split, t_cache_split, c_emb_cache_split, class_cache_split, rank):
        img_cache = torch.empty(0)
        t_cache = torch.empty(0)
        c_emb_cache = torch.empty(0)
        class_cache = torch.empty(0)


        if rank == 0:
            dist.scatter(img_cache, scatter_list=img_cache_split, src=0)
            dist.scatter(t_cache, scatter_list=t_cache_split, src=0)
            dist.scatter(c_emb_cache, scatter_list=c_emb_cache_split, src=0)
            dist.scatter(class_cache, scatter_list=class_cache_split, src=0)
        else:
            dist.scatter(img_cache, src=0)
            dist.scatter(t_cache, src=0)
            dist.scatter(c_emb_cache, src=0)
            dist.scatter(class_cache, src=0)

        return img_cache, t_cache, c_emb_cache, class_cache
    
    def __len__(self):
        # 데이터셋의 길이를 img_cache의 길이로 반환 (모든 캐시는 동일 길이를 가정)
        return len(self.img_cache)
    
    def __getitem__(self, idx):
        # 각 인덱스에 해당하는 데이터를 반환
        img = self.img_cache[idx]
        t = self.t_cache[idx]
        c_emb = self.c_emb_cache[idx]
        class_label = self.class_cache[idx]
        return img, t, c_emb, class_label, idx  # 인덱스도 반환하여 이후 업데이트에 사용
      
    def update_data(self, indices, new_imgs):
        # 이 부분을 고쳐서 `indices`를 정수 배열 형태로 바꿔 인덱싱
        indices = indices.view(-1).long()  # ensure indices are a flat, long tensor
        
        device = self.img_cache.device
        indices = indices.to(device)
        new_imgs = new_imgs.to(device)
        
        # print('self.img_cache device:', self.img_cache.device)
        # print('indices device:', indices.device)
        # print('new_imgs device:', new_imgs.device)
        
        # 인덱스가 맞지 않는 경우 torch.index_select로 인덱스를 처리
        self.img_cache.index_copy_(0, indices, new_imgs)  # indices에 맞는 부분만 교체
        self.t_cache.index_copy_(0, indices, self.t_cache[indices] - 1)
        
        # t_cache 값이 0 미만인 인덱스 처리
        negative_indices = (self.t_cache[indices] < 0).nonzero(as_tuple=True)[0]
        
        # 실제 zero_indices를 전체 t_cache 기준으로 변환
        zero_indices = indices[negative_indices]
        num_zero_indices = zero_indices.size(0)

        if num_zero_indices > 0:
            # 0인 인덱스를 T-1로 초기화
            self.t_cache.index_fill_(0, zero_indices, 999)
            self.img_cache.index_copy_(0, zero_indices, torch.randn(
                num_zero_indices, 
                new_imgs.shape[1],  # channels
                new_imgs.shape[2],  # height
                new_imgs.shape[3],  # width
                device=device 
            ))

def custom_collate_fn(batch):
    # 배치 데이터를 분리하여 텐서로 변환
    imgs, ts, c_embs, class_labels, indices = zip(*batch)
    imgs = torch.stack(imgs)  # 배치 이미지를 스택으로 만듦
    ts = torch.tensor(ts)
    c_embs = torch.stack(c_embs)
    class_labels = torch.tensor(class_labels)
    indices = torch.tensor(indices)
    return imgs, ts, c_embs, class_labels, indices
