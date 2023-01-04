import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf
import hydra

#@click.group()
#def cli():
#    pass


#@click.command()
#@click.option("--lr", default=1e-3, help='learning rate to use for training')

@hydra.main(config_path='config', config_name="config.yaml", version_base=None)


def train(config):
    """Train the model on the training data and return the obtained weights"""
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config['hyperparameters']
    print("Training day and night")
    #print(lr)

    model = MyAwesomeModel()
    train_set = torch.load('data/processed/train.data')


    trainset = TensorDataset(train_set['images'], train_set['labels'])

    # Download and load the training data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])

    epochs = hparams["n_epochs"]
    loss_for_plot = []
    for e in range(epochs):
        running_loss = 0
        print(f'Beginning the #',e, 'epoch')
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps, _ = model(images)
            loss = criterion(log_ps, labels.to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('Loss = ',running_loss)
        loss_for_plot.append(running_loss)
    # Save model
    path = "models/checkpoint.pth"
    torch.save(model.state_dict(), path) 
    print('Model is saved!')
    # Make plot
    plt.plot(range(epochs),loss_for_plot)
    plt.savefig('reports/figures/trainloss.png')
    print('Plot is saved as reports/figures/trainloss.png')


#cli.add_command(train)

#@click.command()
#@click.argument("model_checkpoint")


if __name__ == "__main__":
    train()
    # path= /Users/carlschmidt/Desktop/dtu_mlops/s1_development_environment/exercise_files/final_exercise
    # python main.py train --lr=0.001
    # python main.py evaluate checkpoint.pth

