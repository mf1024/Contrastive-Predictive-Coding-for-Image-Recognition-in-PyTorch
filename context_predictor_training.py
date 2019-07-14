import torch
import random
import itertools
import datetime
import os

from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import cos_loss, norm_euclidian, get_random_patches, get_patch_tensor_from_image_batch

def run_context_predictor(args, res_encoder_model, context_predictor_model, models_store_path):

    print("RUNNING CONTEXT PREDICTOR TRAINING")

    dataset_train, dataset_test = get_imagenet_datasets(args.image_folder, num_classes = args.num_classes)

    def get_random_patch_loader():
        return DataLoader(dataset_train, args.num_random_patches, shuffle=True)

    random_patch_loader = get_random_patch_loader()
    data_loader_train = DataLoader(dataset_train, args.sub_batch_size, shuffle = True)

    params = list(res_encoder_model.parameters()) + list(context_predictor_model.parameters())
    optimizer = torch.optim.Adam(params = params, lr=0.0005)

    sub_batches_processed = 0
    batch_loss = 0
    best_batch_loss = 1e10

    z_vect_similarity = dict()

    for batch in data_loader_train:

        # plt.imshow(img_arr.permute(1,2,0))
        # fig, axes = plt.subplots(7,7)

        img_batch = batch['image'].to(args.device)
        patch_batch = get_patch_tensor_from_image_batch(img_batch)

        patches_encoded = res_encoder_model.forward(patch_batch)
        patches_encoded = patches_encoded.view(img_batch.shape[0], 7,7,-1)
        patches_encoded = patches_encoded.permute(0,3,1,2)

        for i in range(2):
            patches_return = get_random_patches(random_patch_loader, args.num_random_patches)
            if patches_return['is_data_loader_finished']:
                random_patch_loader = get_random_patch_loader()
            else:
                random_patches = patches_return['patches_tensor'].to(args.device)

        # enc_random_patches = resnet_encoder.forward(random_patches).detach()
        enc_random_patches = res_encoder_model.forward(random_patches)

        predictions = context_predictor_model.forward(patches_encoded)
        losses = []

        for p in predictions:

            target = patches_encoded[:,:,p['y'],p['x']]
            pred = p['prediction']
            pred_class = p['pred_class']

            # print(f"pred.shape {pred.shape}")
            # print(f"target.shape {target.shape}")

            cos_loss_val = cos_loss(pred.detach().to('cpu'), target.detach().to('cpu'))
            euc_loss_val = norm_euclidian(pred.detach().to('cpu'), target.detach().to('cpu'))

            if pred_class in z_vect_similarity:
                z_vect_similarity[pred_class] = torch.cat([z_vect_similarity[pred_class], cos_loss_val], dim = 0)
                # print(f"gut cos_sim {cos_loss_val}")
                # print(f"gut mse: {euc_loss_val}")
            else:
                z_vect_similarity[pred_class] = cos_loss_val

            #print(f"target shape {target.shape} pred shape {pred.shape}")

            good_term = cos_loss(pred, target)
            divisor = cos_loss(pred, target)

            store_similarity_idx = random.randint(0, args.num_random_patches-1)

            for random_patch_idx in range(args.num_random_patches):
                divisor = divisor + cos_loss(pred, enc_random_patches[random_patch_idx:random_patch_idx+1])

                if random_patch_idx == store_similarity_idx:

                    pred_class = pred_class+"b"

                    cos_loss_val = cos_loss(pred.detach().detach().to('cpu'), enc_random_patches[random_patch_idx:random_patch_idx+1].detach().to('cpu'))
                    euc_loss_val = norm_euclidian(pred.detach().detach().to('cpu'), enc_random_patches[random_patch_idx:random_patch_idx+1].detach().to('cpu'))

                    if pred_class in z_vect_similarity:
                        z_vect_similarity[pred_class] = torch.cat([z_vect_similarity[pred_class], cos_loss_val], dim = 0)
                        # print(f"bad cos_sim {cos_loss_val}")
                        # print(f"bad mse: {euc_loss_val}")
                    else:
                        z_vect_similarity[pred_class] = cos_loss_val


            losses.append(-torch.log(good_term/divisor))

        loss = torch.sum(torch.cat(losses))
        loss.backward()

        sub_batches_processed += img_batch.shape[0]
        batch_loss += loss.detach().to('cpu')

        if sub_batches_processed >= args.batch_size:

            optimizer.step()
            optimizer.zero_grad()
            print(f"{datetime.datetime.now()} Loss: {batch_loss}")

            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder_weights.pt"))
            torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "last_context_predictor_weights.pt"))

            if best_batch_loss > batch_loss:
                best_batch_loss = batch_loss
                torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
                torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "best_context_predictor_weights.pt"))

            for key, cos_similarity_tensor in z_vect_similarity.items():
                print(f"Mean cos_sim for class {key} is {cos_similarity_tensor.mean()} . Number: {cos_similarity_tensor.size()}")

            z_vect_similarity = dict()

            sub_batches_processed = 0
            batch_loss = 0
