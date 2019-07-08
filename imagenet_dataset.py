from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from skimage import io
import numpy as np
import time
from PIL import Image

IMG_SIZE = (256,256)

class ImageNetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = 0.9, random_seed = 42, target_transform = None, num_classes = None):
        super(ImageNetDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        for class_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path,class_name)):
                continue
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.ToTensor()
        img1 = tr(img)

        width, height = img.size
        if min(width, height)>IMG_SIZE[0] * 1.5:
            tr = transforms.Resize(int(IMG_SIZE[0] * 1.5))
            img = tr(img)

        width, height = img.size
        if min(width, height)<IMG_SIZE[0]:
            tr = transforms.Resize(IMG_SIZE)
            img = tr(img)

        tr = transforms.RandomCrop(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)

        if (img.shape[0] != 3):
            img = img[0:3]

        return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']


def get_imagenet_datasets(data_path, num_classes = None):

    random_seed = int(time.time())

    dataset_train = ImageNetDataset(data_path,is_train = True, random_seed=random_seed, num_classes = num_classes)
    dataset_test = ImageNetDataset(data_path, is_train = False, random_seed=random_seed, num_classes = num_classes)

    return dataset_train, dataset_test

#
# data_path = "/Users/martinsf/data/images_1/imagenet_images/"
# dataset_train, dataset_test = get_imagenet_datasets(data_path)
#
# print(f"Number of train samplest {dataset_train.__len__()}")
# print(f"Number of samples in test split {dataset_test.__len__()}")
#
# BATCH_SIZE = 200
#
# data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
# data_loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = True)
#
#
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(BATCH_SIZE//20,20, figsize=(6,10))
#
# for batch in data_loader_train:
#
#     print(f"Shape of batch['image'] {batch['image'].shape}")
#     print(f"Shape of batch['cls'] {batch['cls'].shape}")
#
#     for i in range(BATCH_SIZE):
#
#         col = i % 20
#         row = i // 20
#
#         img = batch['image'][i].numpy()
#
#         axes[row,col].set_axis_off()
#         #axes[row,col].set_title(batch['class_name'][i])
#         axes[row,col].imshow(np.transpose(img,(1,2,0)))
#
#     plt.show()
#
#     break


