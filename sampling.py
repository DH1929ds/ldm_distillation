import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from absl import app, flags
import warnings


    
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
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


def sampling():
    model = get_model()
    sampler = DDIMSampler(model)
    
    classes = [0, 1, 2, 3]  # define classes to be sampled here
    n_samples_per_class = 8
    
    ddim_steps = 100
    ddim_eta = 1.0
    scale = 1.5 # for unconditional guidance 0: uncond, 1: no guidance cond
    ddim_use_original_steps = True
    
    all_samples = list()
    
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
            )
    
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 ddim_use_original_steps=ddim_use_original_steps,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)
            
    
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)
    
    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)
    
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    
    # save as image
    output_image = Image.fromarray(grid.astype(np.uint8))
    output_image.save('output.png')  # 파일로 저장

def sampling_with_intermediates(batch_size=32):
    model = get_model()

    # 모델을 여러 GPU에 분산
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)  # 모델을 여러 GPU로 분산
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    sampler = DDIMSampler(model)
    
    classes = list(range(1001))  # Define classes to be sampled here
    n_samples_per_class = 6
    
    ddim_steps = 100
    ddim_eta = 1.0
    scale = 1  # for unconditional guidance 0: uncond, 1: no guidance cond
    ddim_use_original_steps = True

    # Dictionaries to store features and condition vectors
    features_by_index = {}
    labels_by_index = {}
    condition_vectors = []  # To store condition vectors 'c'
    condition_labels = []   # To store class labels for condition vectors 'c'

    with torch.no_grad():
        # DataParallel을 사용하는 경우와 사용하지 않는 경우 구분
        ema_scope = model.module.ema_scope if hasattr(model, "module") else model.ema_scope
        
        with ema_scope():  # DataParallel일 경우 model.module을 사용
            # Process classes in batches
            for i in range(0, len(classes), batch_size):
                class_batch = classes[i:i + batch_size]
                print(f"Rendering {n_samples_per_class} examples for classes {class_batch} in {ddim_steps} steps and using s={scale:.2f}.")

                # Create conditioning for a batch of classes
                xc = torch.tensor([label for class_label in class_batch for label in [class_label] * n_samples_per_class])
                c = model.module.get_learned_conditioning({model.module.cond_stage_key: xc.to(model.device)}) if hasattr(model, "module") else model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                # Create unconditional conditioning 'uc' for the current batch size
                uc = model.module.get_learned_conditioning({model.module.cond_stage_key: torch.tensor(len(xc) * [1000]).to(model.device)}) if hasattr(model, "module") else model.get_learned_conditioning({model.cond_stage_key: torch.tensor(len(xc) * [1000]).to(model.device)})

                # Store condition vectors 'c'
                condition_vectors.extend(c.cpu().numpy())  # Collecting condition vectors
                condition_labels.extend([class_label for class_label in class_batch for _ in range(n_samples_per_class)])  # Collecting corresponding class labels

                samples_ddim, intermediates = sampler.sample_with_intermediates_features(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=len(xc),
                    shape=[3, 64, 64],
                    log_every_t=100,
                    verbose=False,
                    ddim_use_original_steps=ddim_use_original_steps,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta
                )

                # Extracting and saving features
                features = intermediates['features']  # Assuming it's a list of length 10
                for index, feature in enumerate(features):
                    if index not in features_by_index:
                        features_by_index[index] = []
                        labels_by_index[index] = []

                    features_by_index[index].extend([f.cpu().numpy() for f in feature])  # Append numpy array of each feature
                    labels_by_index[index].extend([class_label for class_label in class_batch for _ in range(n_samples_per_class)])

    # Save features and labels as numpy arrays for each index for later use
    for index in features_by_index:
        np.save(f'features_index_{index}.npy', np.array(features_by_index[index]))
        np.save(f'labels_index_{index}.npy', np.array(labels_by_index[index]))

    # Save condition vectors and labels
    np.save('condition_vectors.npy', np.array(condition_vectors))
    np.save('condition_labels.npy', np.array(condition_labels))

    for index in range(len(intermediates['features'])):
        tsne_visualization_by_index(index)
    
    # Perform t-SNE on condition vectors
    tsne_visualization_conditions('condition_vectors.npy', 'condition_labels.npy')


def tsne_visualization_by_index(index):
    # Define the folder path
    save_dir = 'tsne_visualization'
    
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load features and labels for the given index
    features = np.load(f'features_index_{index}.npy')
    labels = np.load(f'labels_index_{index}.npy')
    
    # Flatten features if necessary, depending on the shape
    features = features.reshape(features.shape[0], -1)  # Flatten to (n_samples, n_features)
    
    # Fit and transform with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(set(labels))))  # Adjust ticks based on the number of unique labels
    plt.title(f"t-SNE of Features for Index {index}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the plot in the tsne_visualization folder
    plt.savefig(os.path.join(save_dir, f'tsne_index_{index}.png'))  # Save figure to file
    plt.close()  # Close the figure to free memory

def tsne_visualization_conditions(conditions_path, labels_path):
    # Define the folder path
    save_dir = 'tsne_visualization'
    
    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load condition vectors and labels
    conditions = np.load(conditions_path)
    labels = np.load(labels_path)
    
    # Flatten condition vectors if necessary, depending on the shape
    conditions = conditions.reshape(conditions.shape[0], -1)  # Flatten to (n_samples, n_features)
    
    # Fit and transform with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    conditions_tsne = tsne.fit_transform(conditions)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(conditions_tsne[:, 0], conditions_tsne[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(set(labels))))  # Adjust ticks based on the number of unique labels
    plt.title("t-SNE of Condition Vectors")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the plot in the tsne_visualization folder
    plt.savefig(os.path.join(save_dir, 'tsne_conditions.png'))  # Save figure to file
    plt.close()  # Close the figure to free memory


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # distill_caching_base()
    sampling_with_intermediates()
    
    # print(torch.cuda.device_count())  # 시스템에서 사용 가능한 GPU 수를 출력
    # print(torch.cuda.current_device())  # 현재 사용 중인 GPU 장치 번호를 출력


if __name__ == '__main__':
    app.run(main)
