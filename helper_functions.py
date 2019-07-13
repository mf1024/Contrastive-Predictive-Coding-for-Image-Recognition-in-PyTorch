import torch
import os
import random
import csv
from torch.utils.data import DataLoader

def cos_loss(a,b):
    dot = torch.sum(a * b, dim=1)
    aa = torch.sum((a**2),dim=1)**0.5
    bb = torch.sum((b**2),dim=1)**0.5
    dot_norm = dot/(aa*bb)
    ret = torch.exp(dot_norm)
    return ret

def norm_euclidian(a,b):
    aa = (torch.sum((a**2),dim=1)**0.5).unsqueeze(dim=1)
    bb = (torch.sum((b**2),dim=1)**0.5).unsqueeze(dim=1)
    return (torch.sum(((a/aa-b/bb)**2),dim=1)**0.5)

def inspect_model(model):
    param_count = 0
    for param_tensor_str in model.state_dict():
        tensor_size = model.state_dict()[param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size} = {model.state_dict()[param_tensor_str].numel()} params")
        param_count += model.state_dict()[param_tensor_str].numel()

    print(f"Number of parameters: {param_count}")


def get_next_model_folder(prefix, path = ''):

    model_folder = lambda prefix, run_idx: f"{prefix}_model_run_{run_idx}"

    run_idx = 1
    while os.path.isdir(os.path.join(path, model_folder(prefix, run_idx))):
        run_idx += 1

    model_path = os.path.join(path, model_folder(prefix, run_idx))
    print(f"STARTING {prefix} RUN {run_idx}! Storing the models at {model_path}")

    return model_path


def get_random_patches(random_patch_loader, num_random_patches):

        is_data_loader_finished = False

        try:
            img_batch = next(iter(random_patch_loader))['image']
        except StopIteration:
            is_data_loader_finished = True
            # random_patch_loader = DataLoader(dataset_train, num_random_patches, shuffle=True)

        if len(img_batch) < num_random_patches:
            is_data_loader_finished = True

        patches = []

        for i in range(num_random_patches):
            x = random.randint(0,6)
            y = random.randint(0,6)

            patches.append(img_batch[i:i+1,:,x*32:x*32+64,y*32:y*32+64])

            # plt.imshow(np.transpose(patches[-1][0],(1,2,0)))
            # plt.show()

        patches_tensor = torch.cat(patches, dim=0)

        return dict(
            patches_tensor = patches_tensor,
            is_data_loader_finished = is_data_loader_finished)


def get_patch_tensor_from_image_batch(img_batch):

    # Input of the function is a tensor [B, C, H, W]
    # Output of the functions is a tensor [B * 49, C, 64, 64]

    patch_batch = None
    all_patches_list = []

    for y_patch in range(7):
        for x_patch in range(7):

            y1 = y_patch * 32
            y2 = y1 + 64

            x1 = x_patch * 32
            x2 = x1 + 64

            img_patches = img_batch[:,:,y1:y2,x1:x2] # Batch(img_idx in batch), channels xrange, yrange
            img_patches = img_patches.unsqueeze(dim=1)
            all_patches_list.append(img_patches)

            # print(patch_batch.shape)
    all_patches_tensor = torch.cat(all_patches_list, dim=1)

    patches_per_image = []
    for b in range(all_patches_tensor.shape[0]):
        patches_per_image.append(all_patches_tensor[b])

    patch_batch = torch.cat(patches_per_image, dim = 0)
    return patch_batch


def write_csv_stats(csv_path, stats_dict):

    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(stats_dict.keys())

    for key, value in stats_dict.items():
        if isinstance(value, float):
            precision = 0.001
            stats_dict[key] =  ((value / precision ) // 1.0 ) * precision

    with open(csv_path, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(stats_dict.values())

