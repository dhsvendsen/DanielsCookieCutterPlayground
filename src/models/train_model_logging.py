import logging

import hydra
import matplotlib.pyplot as plt
import torch
from model import ConvNet
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    log.info("Training day and night")

    # TODO: Implement training loop here
    # Pytorch train and test sets
    # Model needs shape (n, 1, 28, 28)
    images = torch.unsqueeze(
        torch.load(f"{hparams['datapath']}/train_images.pt"), dim=1
    )
    labels = torch.load(f"{hparams['datapath']}/train_labels.pt")
    train = TensorDataset(images, labels)
    train_set = DataLoader(train, batch_size=hparams["batch_size"], shuffle=True)

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    losses = []
    for epoch in range(hparams["n_epochs"]):
        running_loss = 0
        for images, labels in train_set:
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, torch.flatten(labels))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_set)
        log.info(f"Epoch [{epoch+1}/{hparams['n_epochs']}], Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

    torch.save(model.state_dict(), f"{hparams['modelpath']}/cnn_checkpoint.pth")

    plt.figure()
    plt.plot([i + 1 for i in range(hparams["n_epochs"])], losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{hparams['reportspath']}/figures/convergence.png", dpi=200)


if __name__ == "__main__":
    train()
