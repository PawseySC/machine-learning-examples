import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse


# Program parameters

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
n_epochs = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to the dataset root directory.")
    args = vars(parser.parse_args())
    # training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    training_data = datasets.FashionMNIST(root=args["data"], train=True, download=False, transform=ToTensor())
    test_data = datasets.FashionMNIST(root=args["data"], train=False, download=False, transform=ToTensor())
    #test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)



    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    for epoch in range(n_epochs):
        print(f"\n### Epoch {epoch}/{n_epochs} ###")
        train(train_dataloader, model, loss_fn, optimizer)
