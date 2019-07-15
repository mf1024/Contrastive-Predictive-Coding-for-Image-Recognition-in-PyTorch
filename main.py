import torch

from torch.utils.data import DataLoader
from models import ResEncoderModel, ContextPredictionModel, ResClassificatorModel
from helper_functions import get_next_model_folder, inspect_model, write_csv_stats

from context_predictor_training import run_context_predictor
from classificator_training import run_classificator

import argparse
import os


parser = argparse.ArgumentParser(description='Contrastive predictive coding params')

# mode = 'train_encoder_context_prediction'
# mode = 'train_classificator'
parser.add_argument('-mode', default='train_classificator' , type=str)
parser.add_argument('-image_folder', default='', type=str)
parser.add_argument('-num_classes', default=10, type=int)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-sub_batch_size', default=2, type=int)
parser.add_argument('-num_random_patches', default=15, type=int)
# cpu or cuda
parser.add_argument('-device', default='cuda', type=str)

args, args_other = parser.parse_known_args()

print(f"Running CPC with args {args}")

Z_DIMENSIONS = 1024

stored_models_root_path = "models"
if not os.path.isdir(stored_models_root_path):
    os.mkdir(stored_models_root_path)


if args.mode == 'train_encoder_context_prediction':

    res_encoder_weights_path = None
    context_predictor_weights_path = None

    res_encoder_model = ResEncoderModel().to(args.device)
    context_predictor_model = ContextPredictionModel(in_channels=Z_DIMENSIONS).to(args.device)

    inspect_model(res_encoder_model)
    inspect_model(context_predictor_model)

    model_store_folder = get_next_model_folder('Context_Pred_Training', stored_models_root_path)
    os.mkdir(model_store_folder)

    if res_encoder_weights_path:
        res_encoder_model.load_state_dict(torch.load(res_encoder_weights_path))

    if context_predictor_weights_path:
        context_predictor_model.load_state_dict(torch.load(context_predictor_weights_path))

    run_context_predictor(args, res_encoder_model, context_predictor_model, model_store_folder)


if args.mode == 'train_classificator':

    res_encoder_weights_path = None
    res_classificator_weights_path = None

    res_encoder_model = ResEncoderModel().to(args.device)
    res_classificator_model = ResClassificatorModel(in_channels=Z_DIMENSIONS, num_classes=args.num_classes).to(args.device)

    inspect_model(res_encoder_model)
    inspect_model(res_classificator_model)

    model_store_folder = get_next_model_folder('Classification_Training', stored_models_root_path)
    os.mkdir(model_store_folder)

    if res_encoder_weights_path:
        res_encoder_model.load_state_dict(torch.load(res_encoder_weights_path))

    if res_classificator_weights_path:
        res_classificator_model.load_state_dict(torch.load(res_classificator_weights_path))

    run_classificator(args, res_classificator_model, res_encoder_model, model_store_folder)

