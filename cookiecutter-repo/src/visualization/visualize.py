import argparse
import sys
import torch
import click
from torch import nn

from torch.utils.data import TensorDataset
from torch.optim import Adam
from sklearn.manifold import TSNE


from model import MyAwesomeModel


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

    tsne = TSNE(n_components=2)

    with torch.no_grad():
        for images, labels in testloader:
            probs, last_linear = model.forward(images)
            print(type(last_linear))
            #top_p, top_class = probs.topk(1, dim=1)
            #equals = top_class == labels.view(*top_class.shape)

            #accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    #accuracy = accuracy / len(testloader)
    #print(f'Accuracy: {accuracy * 100}%')


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    # path= /Users/carlschmidt/Desktop/dtu_mlops/s1_development_environment/exercise_files/final_exercise
    # python main.py train --lr=0.001
    # python main.py evaluate checkpoint.pth

