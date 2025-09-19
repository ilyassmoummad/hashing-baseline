import argparse


parser = argparse.ArgumentParser()

# General
parser.add_argument("--device", type=str, default='cuda') #device for inference
parser.add_argument("--save", action='store_true') #save results
parser.add_argument("--results_dir", type=str, default='') #directory to save results

# Data
parser.add_argument("--dataset", type=str, default='esc50') #dataset to use ['speechcommands', 'vocalsound', 'gtzan', 'esc50']
parser.add_argument("--data_dir", type=str, default='') #directory of the dataset
parser.add_argument("--n_workers", type=int, default=16) #number of workers for dataloader
parser.add_argument("--bs", type=int, default=256) #batch size

# Model
parser.add_argument("--model", type=str, default='dasheng') #model for retrieval ['dasheng', 'clap', 'whisper', 'ced']
parser.add_argument("--pretrained_model", type=str, default='') #path to pretrained weights for the model
parser.add_argument("--bits", type=int, default=16) #number of bits for retrieval

# Evaluation
parser.add_argument("--n_runs", type=int, default=10) #number of evaluation runs
parser.add_argument("--knn_k", type=int, default=200) #nearest neighbors for kNN
parser.add_argument("--knn_t", type=float, default=0.07) #kNN temperature
parser.add_argument("--nopca", action='store_true') #use original features

args = parser.parse_args()