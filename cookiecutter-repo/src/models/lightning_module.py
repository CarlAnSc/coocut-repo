from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from torch import nn, optim, load
import torch
from torch.utils.data import DataLoader



"""
The following should be used for a convolutional NN.
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU()
            """

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 10),
        )

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

        self.criterion = nn.CrossEntropyLoss() #NLLLoss()

        self.train_dataset = load('data/processed/train.data')

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.backbone(x)
            
        last_linear = x

        # output so no dropout here
        x = F.log_softmax(x, dim=1)

        return x, last_linear 
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds, lastlinear = self(data)
        #print(preds)
        loss = self.criterion(preds, target)
        return loss 

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=10)

if __name__ == "__main__":
    model = MyAwesomeModel()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model)