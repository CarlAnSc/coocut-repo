import argparse
import sys
import torch
import click
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import TensorDataset
from torch.optim import Adam
from sklearn.manifold import TSNE

from src.models.model import MyAwesomeModel


@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    print("Visualizing final data features found by model")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)


    data = torch.load('data/processed/train.data')
    dataset = TensorDataset(data['images'], data['labels'])

    # Download and load the testing data
    testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    model.eval()

    data_for_plot = []
    label_for_plot = []
    with torch.no_grad():
        for images, labels in testloader:

            probs, last_linear = model.forward(images)
            data_for_plot.append(last_linear.numpy())
            label_for_plot.append(labels.numpy())

    data_for_plot = np.concatenate(data_for_plot)
    label_for_plot = np.concatenate(label_for_plot)

    print('Running TSNE-transform:')
    tsne = TSNE(n_components=2)
    transData = tsne.fit_transform(data_for_plot)

    print('Sending plot to reports/figures/TSNE-visualization')
    plt.figure()
    plt.scatter(transData[:,0],transData[:,1],c=label_for_plot)
    plt.savefig('reports/figures/TSNE-visualization.png')
    

cli.add_command(visualize)


if __name__ == "__main__":
    cli()
    # path= /Users/carlschmidt/Desktop/dtu_mlops/s1_development_environment/exercise_files/final_exercise
    # python main.py train --lr=0.001
    # python main.py evaluate checkpoint.pth

