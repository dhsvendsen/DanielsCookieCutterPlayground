import hydra
import torch
import wandb
from model import ConvNet
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
def train(cfg):

    # Wandb init
    wandb.init(
        # Set the project where this run will be logged
        project=cfg.wandb["project"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=cfg.wandb["name"],
    )

    wandb.config = OmegaConf.to_container(
        cfg.experiment, resolve=True, throw_on_missing=True
    )

    wandb.log(wandb.config)

    print("Training day and night")
    print(wandb.config)

    # TODO: Implement training loop here
    # Pytorch train and test sets
    # Model needs shape (n, 1, 28, 28)
    images = torch.unsqueeze(
        torch.load(f"{wandb.config['datapath']}/train_images.pt"), dim=1
    )
    labels = torch.load(f"{wandb.config['datapath']}/train_labels.pt")
    train = TensorDataset(images, labels)
    train_set = DataLoader(train, batch_size=wandb.config["batch_size"], shuffle=True)

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])

    # wandb is watching our CNN
    wandb.watch(model, log_freq=100)

    losses = []
    for epoch in range(wandb.config["n_epochs"]):
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

        print(f"Epoch: [{epoch+1}/{wandb.config['n_epochs']}], Loss: {epoch_loss:.4f}")
        wandb.log({"Epoch loss": epoch_loss})
        losses.append(epoch_loss)

    torch.save(model.state_dict(), f"{wandb.config['modelpath']}/cnn_checkpoint.pth")

    # Image prediction logging utility
    def log_image_table(images, predicted, labels):
        "Log a wandb.Table with (img, pred, target, scores)"
        # 🐝 Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["image", "pred", "target"])
        for img, pred, targ in zip(
            images.to("cpu"), predicted.to("cpu"), labels.to("cpu")
        ):
            table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ)
        wandb.log({"predictions_table": table})

    # Get one batch (of training data like a noob) and plot some predictions
    with torch.no_grad():
        accuracy = torch.tensor([0], dtype=torch.float)
        for images, labels in train_set:
            model.to("cpu")
            log_ps = model(images)
            top_p, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            log_image_table(images, top_class, labels)
            break


if __name__ == "__main__":
    train()
