import torch
import itertools
from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import get_patch_tensor_from_image_batch

import os

def run_classificator(res_classificator_model, res_encoder_model, models_store_path, num_classes, device):

    print("RUNNING CLASSIFICATOR TRAINING")

    data_path = "/Users/martinsf/data/images_1/imagenet_images"
    dataset_train, dataset_test = get_imagenet_datasets(data_path, num_classes = num_classes)

    SUB_BATCH_SIZE = 2
    BATCH_SIZE = 2
    EPOCHS = 10
    LABELS_PER_CLASS = 25

    data_loader_train = DataLoader(dataset_train, SUB_BATCH_SIZE, shuffle = True)
    data_loader_test = DataLoader(dataset_test, SUB_BATCH_SIZE, shuffle = True)

    NUM_TRAIN_SAMPLES = dataset_train.get_number_of_samples()
    NUM_TEST_SAMPLES = dataset_test.get_number_of_samples()


    optimizer = torch.optim.Adam(params = itertools.chain(res_classificator_model.parameters(), res_encoder_model.parameters()), lr=0.0005)

    best_epoch_test_loss = 0.0

    for epoch in range(EPOCHS):

        sub_batches_processed = 0

        epoch_train_true_positives = 0.0
        epoch_training_loss = 0.0
        epoch_train_losses = []
        epoch_training_accuracy = 0.0

        for batch in data_loader_train:

            img_batch = batch['image'].to(device)

            patch_batch = get_patch_tensor_from_image_batch(img_batch, SUB_BATCH_SIZE)
            patches_encoded = res_encoder_model.forward(patch_batch)

            patches_encoded = patches_encoded.view(SUB_BATCH_SIZE, 7,7,-1)
            patches_encoded = patches_encoded.permute(0,3,1,2)

            classes = batch['cls'].to(device)

            y_one_hot = torch.zeros(img_batch.shape[0], num_classes).to(device)
            y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            pred = res_classificator_model.forward(patches_encoded)
            loss = torch.sum(-y_one_hot * torch.log(pred))
            epoch_train_losses.append(loss.detach().to('cpu').numpy())
            epoch_training_loss += loss.detach().to('cpu').numpy()

            print(pred.argmax(dim=1))
            print(classes)
            epoch_train_true_positives += torch.sum(pred.argmax(dim=1) == classes)

            loss.backward()

            sub_batches_processed += SUB_BATCH_SIZE


            if sub_batches_processed >= BATCH_SIZE:
                res_classificator_model.eval()
                optimizer.step()
                optimizer.zero_grad()
                sub_batches_processed = 0


        epoch_accuracy = float(epoch_train_true_positives) / float(NUM_TRAIN_SAMPLES)
        print(f"Epoch {epoch} train accuarcy: {epoch_accuracy}")

        print(f"Training loss of epoch {epoch} is {epoch_training_loss}")
        print(f"Accuracy of epoch {epoch} is {epoch_training_accuracy}")


        epoch_test_true_positives = 0.0
        epoch_test_loss = 0.0
        epoch_test_losses = []

        for batch in data_loader_test:

            img_batch = batch['image'].to(device)

            patch_batch = get_patch_tensor_from_image_batch(img_batch, SUB_BATCH_SIZE)
            patches_encoded = res_encoder_model.forward(patch_batch)

            patches_encoded = patches_encoded.view(SUB_BATCH_SIZE, 7,7,-1)
            patches_encoded = patches_encoded.permute(0,3,1,2)

            classes = batch['cls'].to(device)

            y_one_hot = torch.zeros(img_batch.shape[0], num_classes).to(device)
            y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            pred = res_classificator_model.forward(img_batch)
            loss = torch.sum(-y_one_hot * torch.log(pred))
            epoch_test_losses.append(loss.detach().to('cpu').numpy())
            epoch_test_loss += loss.detach().to('cpu')

            epoch_test_true_positives += torch.sum(pred.argmax(dim=1) == classes)
            print(f"Testing epoch: {epoch} accuarcy: {epoch_accuracy}")

        epoch_test_accuracy = float(epoch_test_true_positives) / float(NUM_TEST_SAMPLES)

        print(f"Test loss of epoch {epoch} is {epoch_test_loss}")
        print(f"Test accuracy of epoch {epoch} is {epoch_test_accuracy}")

        res_classificator_model.train()

        torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder.pt"))
        torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "last_res_classificator_model.pt"))

        if best_epoch_test_loss > epoch_test_loss:

            best_epoch_test_loss = epoch_test_loss
            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder.pt"))
            torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "best_res_classificator_model.pt"))

