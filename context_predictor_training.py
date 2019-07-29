import torch
import datetime
import os

from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import dot, dot_norm, dot_norm_exp, norm_euclidian, get_random_patches, get_patch_tensor_from_image_batch
from helper_functions import write_csv_stats

def run_context_predictor(args, res_encoder_model, context_predictor_model, models_store_path):

    print("RUNNING CONTEXT PREDICTOR TRAINING")

    stats_csv_path = os.path.join(models_store_path, "pred_stats.csv")

    dataset_train, dataset_test = get_imagenet_datasets(args.image_folder, num_classes = args.num_classes)

    def get_random_patch_loader():
        return DataLoader(dataset_train, args.num_random_patches, shuffle=True)

    random_patch_loader = get_random_patch_loader()
    data_loader_train = DataLoader(dataset_train, args.sub_batch_size, shuffle = True)

    params = list(res_encoder_model.parameters()) + list(context_predictor_model.parameters())
    optimizer = torch.optim.Adam(params = params, lr=0.00001)

    sub_batches_processed = 0
    batch_loss = 0
    sum_batch_loss = 0 
    best_batch_loss = 1e10

    z_vect_similarity = dict()

    for batch in data_loader_train:

        # plt.imshow(img_arr.permute(1,2,0))
        # fig, axes = plt.subplots(7,7)

        img_batch = batch['image'].to(args.device)
        patch_batch = get_patch_tensor_from_image_batch(img_batch)
        batch_size = len(img_batch)

        patches_encoded = res_encoder_model.forward(patch_batch)
        patches_encoded = patches_encoded.view(batch_size, 7,7,-1)
        patches_encoded = patches_encoded.permute(0,3,1,2)

        for i in range(2):
            patches_return = get_random_patches(random_patch_loader, args.num_random_patches)
            if patches_return['is_data_loader_finished']:
                random_patch_loader = get_random_patch_loader()
            else:
                random_patches = patches_return['patches_tensor'].to(args.device)

        # enc_random_patches = resnet_encoder.forward(random_patches).detach()
        enc_random_patches = res_encoder_model.forward(random_patches)

        # TODO: vectorize the context_predictor_model - stack all 3x3 contexts together
        predictions, locations = context_predictor_model.forward(patches_encoded)
        losses = []

        for b in range(len(predictions)//batch_size):

            b_idx_start = b*batch_size
            b_idx_end = (b+1)*batch_size

            p_y = locations[b_idx_start][0]
            p_x = locations[b_idx_start][1]

            target = patches_encoded[:,:,p_y,p_x]
            pred = predictions[b_idx_start:b_idx_end]

            dot_norm_val = dot_norm_exp(pred.detach().to('cpu'), target.detach().to('cpu'))
            euc_loss_val = norm_euclidian(pred.detach().to('cpu'), target.detach().to('cpu'))

            good_term_dot = dot(pred, target)
            dot_terms = [torch.unsqueeze(good_term_dot,dim=0)]

            for random_patch_idx in range(args.num_random_patches):

                bad_term_dot = dot(pred, enc_random_patches[random_patch_idx:random_patch_idx+1])
                dot_terms.append(torch.unsqueeze(bad_term_dot, dim=0))

            log_softmax = torch.log_softmax(torch.cat(dot_terms, dim=0), dim=0)
            losses.append(-log_softmax[0,])
            # losses.append(-torch.log(good_term/divisor))

        loss = torch.mean(torch.cat(losses))
        loss.backward()

        # loss = torch.sum(torch.cat(losses))
        # loss.backward()

        sub_batches_processed += img_batch.shape[0]
        batch_loss += loss.detach().to('cpu')
        sum_batch_loss += torch.sum(torch.cat(losses).detach().to('cpu'))

        if sub_batches_processed >= args.batch_size:

            optimizer.step()
            optimizer.zero_grad()

            print(f"{datetime.datetime.now()} Loss: {batch_loss}")
            print(f"{datetime.datetime.now()} SUM Loss: {sum_batch_loss}")

            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder_weights.pt"))
            torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "last_context_predictor_weights.pt"))

            if best_batch_loss > batch_loss:
                best_batch_loss = batch_loss
                torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
                torch.save(context_predictor_model.state_dict(), os.path.join(models_store_path, "best_context_predictor_weights.pt"))

            for key, cos_similarity_tensor in z_vect_similarity.items():
                print(f"Mean cos_sim for class {key} is {cos_similarity_tensor.mean()} . Number: {cos_similarity_tensor.size()}")

            z_vect_similarity = dict()


            stats = dict(
                batch_loss = batch_loss,
                sum_batch_loss = sum_batch_loss
            )
            write_csv_stats(stats_csv_path, stats)

            sub_batches_processed = 0
            batch_loss = 0
            sum_batch_loss = 0

