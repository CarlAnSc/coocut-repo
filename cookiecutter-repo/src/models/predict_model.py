import argparse
import sys

import click
import torch
from model import MyAwesomeModel
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset


@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)


    test_set = torch.load('data/processed/test.data')
    testset = TensorDataset(test_set['images'], test_set['labels'])

    # Download and load the testing data
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    model.eval()

    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            probs, _ = model.forward(images)

            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    accuracy = accuracy / len(testloader)
    print(f'Accuracy: {accuracy * 100}%')


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    # path= /Users/carlschmidt/Desktop/dtu_mlops/s1_development_environment/exercise_files/final_exercise
    # python main.py train --lr=0.001
    # python main.py evaluate checkpoint.pth

