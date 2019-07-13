import torch

from torch.utils.data import DataLoader
from models import ResEncoderModel, ContextPredictionModel, ResClassificatorModel
from helper_functions import get_next_model_folder, inspect_model

from context_predictor_training import run_context_predictor
from classificator_training import run_classificator

import os

# mode = 'train_encoder_context_prediction'
mode = 'train_classificator'

DEVICE = 'cpu'
Z_DIMENSIONS = 1024
NUM_CLASSES = 100


stored_models_root_path = "models"
if not os.path.isdir(stored_models_root_path):
    os.mkdir(stored_models_root_path)


if mode == 'train_encoder_context_prediction':

    res_encoder_weights_path = None
    context_predictor_weights_path = None

    res_encoder_model = ResEncoderModel().to(DEVICE)
    context_predictor_model = ContextPredictionModel(in_channels=Z_DIMENSIONS).to(DEVICE)

    inspect_model(res_encoder_model)
    inspect_model(context_predictor_model)

    model_store_folder = get_next_model_folder('Context_Pred_Training', stored_models_root_path)
    os.mkdir(model_store_folder)

    if res_encoder_weights_path:
        res_encoder_model.load_state_dict(torch.load(res_encoder_weights_path))

    if context_predictor_weights_path:
        context_predictor_model.load_state_dict(torch.load(context_predictor_weights_path))

    run_context_predictor(res_encoder_model, context_predictor_model, model_store_folder, NUM_CLASSES, DEVICE)


if mode == 'train_classificator':

    res_encoder_weights_path = None
    res_classificator_weights_path = None

    res_encoder_model = ResEncoderModel().to(DEVICE)
    res_classificator_model = ResClassificatorModel(in_channels=Z_DIMENSIONS, num_classes=NUM_CLASSES).to(DEVICE)

    inspect_model(res_encoder_model)
    inspect_model(res_classificator_model)

    model_store_folder = get_next_model_folder('Classification_Training', stored_models_root_path)
    os.mkdir(model_store_folder)

    if res_encoder_weights_path:
        res_encoder_model.load_state_dict(torch.load(res_encoder_weights_path))

    if res_classificator_weights_path:
        context_predictor_model.load_state_dict(torch.load(res_classificator_weights_path))

    run_classificator(res_classificator_model, res_encoder_model, model_store_folder, NUM_CLASSES, DEVICE)


# TODO Training scheduling - when it converges - try to add more negative samples, or try changing learning rate
# TODO: How to use batch norm correctly when collecting the gradients from smaller batches of two??
# TODO: Data augmentation for patches, to remove simple cues for predicting following images - like straight lines and gradually changing colors.
# TODO: Write custom PyTorch dataset that will prepare unlabeled dataset and labeled dataset for Semi-supervised settings
# TODO: labeled, unlabeled, training = get_imgaenet_semi_supervised_learning_datasets()
