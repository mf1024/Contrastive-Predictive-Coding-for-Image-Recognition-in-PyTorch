import torch
from torch import nn
from torch.nn import Module
from resnet_blocks import ResNetBottleneckBlock
from imagenet_dataset import get_imagenet_datasets

from torch.utils.data import DataLoader

import itertools
import random

import numpy as np
import matplotlib.pyplot as plt

import os
import datetime

DEVICE = 'cuda'

class ResnetEncoder(Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()

        # TODO: ResNet v2
        # TODO: Squeeze and Excitation

        # Input is 3 x 64 x 64
        # prep -> 256 x 32 x 32

        self.conv_blocks = [10,10,10] #256x32x32 -> 512x16x16 -> 1024x8x8
        self.num_blocks = len(self.conv_blocks)
        self.start_channels = 256


        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = self.start_channels,
                kernel_size = 7,
                stride = 1, #Let's not reduce twice
                padding = 3
            ),
            nn.BatchNorm2d(
                num_features = self.start_channels,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )
        # Output 256x 32 x 32
        current_channels = self.start_channels

        self.resnet_blocks = nn.ModuleList()

        for block_idx, conv_block_num in enumerate(self.conv_blocks):
            resnet_block = nn.Sequential()

            for conv_block_idx in range(conv_block_num):

                is_downsampling_block = False

                if block_idx > 0 and conv_block_idx == 0:
                    is_downsampling_block = True

                resnet_block.add_module(
                    f'conv_{conv_block_idx}',
                    ResNetBottleneckBlock(
                        in_channels_block = current_channels,
                        is_downsampling_block = is_downsampling_block
                    )
                )

                if is_downsampling_block:
                    current_channels *= 2

            self.resnet_blocks.append(resnet_block)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self, x):

        x = self.prep.forward(x)
        #print(f'shape after prep {x.shape}')
        for i in range(self.num_blocks):
            x = self.resnet_blocks[i].forward(x)
            #print(f'shape after resnet_block {i} {x.shape}')

        #print(f'shape after resnet {x.shape}')
        x = self.avg_pool.forward(x)
        ##print(f'shape after avg_pool {x.shape}')

        return x


class ContextPredictionModel(Module):

    def __init__(self, in_channels):
        super(ContextPredictionModel, self).__init__()

        self.in_channels = in_channels

        # Input will be 1024x7x7

        # Two sets of convolutional context networks - one for vertical, one for horizontal agregation.

        # Prediction 3 steps ahead. So I will have 8 outputs.
        # [0,2:6] predict->[3,4,5:6],[1,3:6] predict->[4,5,6:6]
        # [4,6:6] predict->[3,2,1:6],[3,5:6] predict->[2,1,0:6]

        # [6:0,2] predict->[6:3,4,5],[6:1,3] predict->[6:4,5,6]
        # [6:4,6] predict->[6:3,2,1],[6:3,5] predict->[6:2,1,0]

        self.context_layers = 3
        self.context_conv = nn.Sequential()

        for layer_idx in range(self.context_layers):
            self.context_conv.add_module(f'batch_norm_{layer_idx}',nn.BatchNorm2d(self.in_channels)),
            self.context_conv.add_module(f'relu_{layer_idx}',nn.ReLU())
            self.context_conv.add_module(
                f'conv2d_{layer_idx}',
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        self.context_conv.add_module(
            'adaptive_avg_pool',
            nn.AdaptiveAvgPool2d(output_size=1)
        )


        # UP - 0, RIGHT = 1, DOWN = 2, LEFT = 3

        self.prediction_weights = nn.ModuleList([nn.ModuleList() for i in range(4)])

        for direction in range(4):
            for prediction_steps in range(3):
                self.prediction_weights[direction].append(
                    nn.Linear(
                        in_features = self.in_channels,
                        out_features = self.in_channels,
                    )
                )


    def forward(self, x):

        prediction_list = []

        for y1 in range(5):
            for x1 in range(5):
                y2 = y1 + 2
                x2 = x1 + 2

                context_input = x[:,:,y1:y2+1,x1:x2+1]
                #print(f"x1:{x1} y1:{y1}, x2:{x2}, y2:{y2}")

                #print(f"context_input shape {context_input.shape}")
                context = self.context_conv.forward(context_input)

                #print(f"context shape {context.shape}")

                context = context.squeeze()
                #print(f"context shape {context.shape}")

                # UP - 0, RIGHT = 1, DOWN = 2, LEFT = 3

                #predict down

                if y2 == 2 or y2 == 3:
                    for steps_y_plus in range(3):
                        y3 = y2 + (steps_y_plus+1)
                        if y3 > 6:
                            break

                        # print(f"y+ pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} y3:{y3}")

                        prediction = self.prediction_weights[0][steps_y_plus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y3,
                                x=x1+1,
                                prediction=prediction,
                                pred_class=f"y+{steps_y_plus+1}",
                            )
                        )


                if y1 == 3 or y1 == 4:
                    for steps_y_minus in range(3):
                        y3 = y1 - (steps_y_minus+1)
                        if y3 < 0:
                            break

                        # print(f"y- pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} y3:{y3}")

                        prediction = self.prediction_weights[1][steps_y_minus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y3,
                                x=x1+1,
                                prediction=prediction,
                                pred_class=f"y-{steps_y_minus+1}",
                            )
                        )


                if x2 == 2 or x2 == 3:
                    for steps_x_plus in range(3):
                        x3 = x2 + (steps_x_plus+1)
                        if x3 > 6:
                            break

                        # print(f"x+ pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} x3:{x3}")
                        prediction = self.prediction_weights[2][steps_x_plus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y1+1,
                                x=x3,
                                prediction=prediction,
                                pred_class=f"x+{steps_x_plus+1}",
                            )
                        )


                if x1 == 3 or x1 == 4:
                    for steps_x_minus in range(3):
                        x3 = x1 - (steps_x_minus+1)
                        if x3 < 0:
                            break

                        # print(f"x- pred for y1:{y1} y2:{y2} x1:{x1} x2{x2} x3:{x3}")
                        prediction = self.prediction_weights[3][steps_x_minus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y1+1,
                                x=x3,
                                prediction=prediction,
                                pred_class=f"x-{steps_x_minus+1}",
                            )
                        )

        return prediction_list


class ClassificationModel(Module):

    def __init__(self):
        super(ClassificationModel, self).__init__()

    def forward(self, encoder_outputs_grid):
        cat = 1
        return cat


NUM_CLASSES = 100
THE_BATCH_SIZE = 16
BATCH_SIZE = 2
data_path = "/home/martin/ai/ImageNet-datasets-downloader/images_4/imagenet_images"
dataset_train, dataset_test = get_imagenet_datasets(data_path, num_classes = NUM_CLASSES)

run_idx = 1
while os.path.isdir(f"models_{run_idx}"):
    run_idx += 1

models_path = f"models_{run_idx}"
os.mkdir(models_path)

print(f"STARTING RUN {run_idx}")

NUM_RANDOM_PATCHES = 15
random_patch_loader = DataLoader(dataset_train, NUM_RANDOM_PATCHES, shuffle=True)

def get_random_patches():

    global random_patch_loader

    try:
        img_batch = next(iter(random_patch_loader))['image']
    except StopIteration:
        random_patch_loader = DataLoader(dataset_train, NUM_RANDOM_PATCHES, shuffle=True)

    if len(img_batch) < NUM_RANDOM_PATCHES:
        random_patch_loader = DataLoader(dataset_train, NUM_RANDOM_PATCHES, shuffle=True)
        img_batch = next(iter(random_patch_loader))['image']


    patches = []

    for i in range(NUM_RANDOM_PATCHES):
        x = random.randint(0,6)
        y = random.randint(0,6)

        patches.append(img_batch[i:i+1,:,x*32:x*32+64,y*32:y*32+64])

        # plt.imshow(np.transpose(patches[-1][0],(1,2,0)))
        # plt.show()

    patches_tensor = torch.cat(patches, dim=0)
    return patches_tensor



data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
data_loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = True)

def inspect_model(model):
    param_count = 0
    for param_tensor_str in model.state_dict():
        tensor_size = model.state_dict()[param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size}")
        param_count += model.state_dict()[param_tensor_str].numel()


    print(f"Number of parameters: {param_count}")

resnet_encoder = ResnetEncoder().to(DEVICE)
context_predictor_model = ContextPredictionModel(in_channels=1024).to(DEVICE)

encoder_load_model_path = "models_12/last_resnet_ecoder.pt"
if encoder_load_model_path is not None:
    checkpoint = torch.load(encoder_load_model_path)
    resnet_encoder.load_state_dict(checkpoint)

context_predictor_load_model_path = "models_12/last_context_predictor_model.pt"
if context_predictor_load_model_path is not None:
    checkpoint = torch.load(context_predictor_load_model_path)
    context_predictor_model.load_state_dict(checkpoint)


inspect_model(resnet_encoder)
inspect_model(context_predictor_model)

optimizer = torch.optim.Adam(params = itertools.chain(resnet_encoder.parameters(), context_predictor_model.parameters()), lr=0.0001)

def cos_loss(a,b):
    dot = torch.sum(a * b, dim=1)
    aa = torch.sum((a**2),dim=1)**0.5
    bb = torch.sum((b**2),dim=1)**0.5
    dot_norm = dot/(torch.max(aa*bb))
    ret = torch.exp(dot_norm)
    return ret

def mse(a,b):
    return torch.mean((a-b)**2)


torch.autograd.set_detect_anomaly(True)

batches_processed = 0
batch_loss = 0
best_batch_loss = 10000000000

z_vect_similarity = dict()

for batch in data_loader_train:


    # plt.imshow(img_arr.permute(1,2,0))
    # fig, axes = plt.subplots(7,7)

    patch_batch = None
    all_patches_list = []

    for y_patch in range(7):
        for x_patch in range(7):

            y1 = y_patch * 32
            y2 = y1 + 64

            x1 = x_patch * 32
            x2 = x1 + 64

            img_patches = batch['image'].to(DEVICE)[:,:,y1:y2,x1:x2] # Batch(img_idx in batch), channels xrange, yrange
            img_patches = img_patches.unsqueeze(dim=1)
            all_patches_list.append(img_patches)

            # print(patch_batch.shape)
    all_patches_tensor = torch.cat(all_patches_list, dim=1)

    patches_per_image = []
    for b in range(BATCH_SIZE):
        patches_per_image.append(all_patches_tensor[b])

    patch_batch = torch.cat(patches_per_image, dim = 0)

    patches_encoded = resnet_encoder.forward(patch_batch)
    patches_encoded = patches_encoded.squeeze()
    patches_encoded = patches_encoded.view(BATCH_SIZE, 7,7,-1)
    patches_encoded = patches_encoded.permute(0,3,1,2)

    random_patches = get_random_patches().to(DEVICE)
    # enc_random_patches = resnet_encoder.forward(random_patches).detach()
    enc_random_patches = resnet_encoder.forward(random_patches)
    enc_random_patches = enc_random_patches.squeeze()

    #print(f"shape after reshape{patches_encoded.shape}")

    predictions = context_predictor_model.forward(patches_encoded)

    losses = []

    for p in predictions:

        target = patches_encoded[:,:,p['y'],p['x']]
        pred = p['prediction']
        pred_class = p['pred_class']

        # print(f"pred.shape {pred.shape}")
        # print(f"target.shape {target.shape}")

        cos_loss_val = cos_loss(pred.detach().to('cpu'), target.detach().to('cpu'))

        if pred_class in z_vect_similarity:
            z_vect_similarity[pred_class] = torch.cat([z_vect_similarity[pred_class], cos_loss_val], dim = 0)
        else:
            z_vect_similarity[pred_class] = cos_loss_val

        #print(f"target shape {target.shape} pred shape {pred.shape}")

        good_term = cos_loss(pred, target)
        divisor = cos_loss(pred, target)

        store_similarity_idx = random.randint(0,NUM_RANDOM_PATCHES-1)

        for random_patch_idx in range(NUM_RANDOM_PATCHES):
            divisor = divisor + cos_loss(pred, enc_random_patches[random_patch_idx:random_patch_idx+1])

            if random_patch_idx == store_similarity_idx:

                pred_class = pred_class+"b"

                cos_loss_val = cos_loss(pred.detach().detach().to('cpu'), enc_random_patches[random_patch_idx:random_patch_idx+1].detach().to('cpu'))

                if pred_class in z_vect_similarity:
                    z_vect_similarity[pred_class] = torch.cat([z_vect_similarity[pred_class], cos_loss_val], dim = 0)
                else:
                    z_vect_similarity[pred_class] = cos_loss_val

           
            # divisor += cos_loss(pred, enc_random_patches[random_patch_idx:random_patch_idx+1])

        losses.append(-torch.log(good_term/divisor))

    loss = torch.sum(torch.cat(losses))
    loss.backward()

    batches_processed += BATCH_SIZE

    batch_loss += loss.detach().to('cpu')

    if batches_processed >= THE_BATCH_SIZE:
        batches_processed = 0

        optimizer.step()
        optimizer.zero_grad()
        print(f"{datetime.datetime.now()} Loss: {batch_loss}")

        batch_loss = 0

        torch.save(resnet_encoder.state_dict(), os.path.join(models_path, "last_resnet_ecoder.pt"))
        torch.save(context_predictor_model.state_dict(), os.path.join(models_path, "last_context_predictor_model.pt"))

        if best_batch_loss > batch_loss:
            best_batch_loss = batch_loss
            torch.save(resnet_encoder.state_dict(), os.path.join(models_path, "best_resnet_ecoder.pt"))
            torch.save(context_predictor_model.state_dict(), os.path.join(models_path, "best_context_predictor_model.pt"))

        for key, cos_similarity_tensor in z_vect_similarity.items():
            print(f"Mean cos_sim for class {key} is {cos_similarity_tensor.mean()} . Number: {cos_similarity_tensor.size()}")

        z_vect_similarity = dict()



#Training the encoder network with the prediction task

#   For image in data

#   TRAINING OF THE ENCODER/

#   keep some pool for random patches

#   prepare 7x7 image crops from the image
#   run the encoder trough each of the images
#   get 7x7 latent vectors for each of the image

#   concat all in Dx7x7 tensor
#   get context outputs from context model from context_model

#   using context outputs make predictions of the other latent vectors

#   get the negative samples from batch
#
#   calculate the losses
#   do the backprop
#   train


#   when the training of the encoder is done -
#   stack the predictor network on top of the encoders and do the prediction


# TODO: Write custom PyTorch dataset that will prepare unlabeled dataset and labeled dataset for Semi-supervised settings
# labeled, unlabeled, training = get_imgaenet_semi_supervised_learning_datasets()

# TODO do we need to run backprop trough the random patches ???? Probably need to check in paper.
# Also need to check how Torch runs the backprop

# TODO Write classification model to put on top of the context model. 

# TODO Training scheduling - when it converges - try to add more negative samples, or try changing learning rate
# TODO Try to implement SGD ?

# TODO: How to use batch norm correctly when collecting the gradients from smaller batches of two??

# TODO: Data augmentation for patches, to remove simple cues for predicting following images - like straight lines and gradually changing colors.
