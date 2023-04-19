import cv2
from torch import tensor
import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math

def collate_batch(batch):
    targets_list = []
    images_list = []
    metadata_list = []
    for dict_ in batch:
        # append target
        targets_list.append(dict_["targets"])

        # append image after loading it
        images_list.append(dict_["image"])

        # append metadata
        metadata_list.append(dict_["metadata"])
        
    images = torch.stack(images_list)
    targets = torch.stack(targets_list)
    return {
        "metadata": metadata_list,
        'image': images, 
        'targets': targets
        } 


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    # device = "cuda"
    # model_is_training = model.training
    model.eval()
    
    ds = 976 #dataloader.val_dataloader()
    A = [[] for i in range(ds)]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        # for batch in tqdm(dataloader.val_dataloader()):
            # for i, (t,J) in enumerate(batch):
        for i, (target, J) in tqdm(enumerate(dataloader.val_dataloader())):

                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J)

                for j in J:
                    A[i].append(j)
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = 19
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = 19

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall