import click
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf
import hydra

def load_data_func():
    train_set, test_set = torch.load('data/processed/train.data'), torch.load('data/processed/test.data')
    return train_set, test_set


