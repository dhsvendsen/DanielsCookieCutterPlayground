import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import click
import matplotlib.pyplot as plt
from model import ConvNet


@click.group()
def cli():
    pass

# Datapath
main_path = 'data/processed'

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = ConvNet()
    state_dict = torch.load(f'{model_checkpoint}')
    model.load_state_dict(state_dict)
    images = torch.unsqueeze( torch.load(f'{main_path}/test_images.pt'), dim=1)
    labels = torch.load(f'{main_path}/test_labels.pt')
    test = TensorDataset(images,labels)
    test_set = DataLoader(test, batch_size=8, shuffle=True)

    with torch.no_grad():
        accuracy = torch.tensor([0], dtype=torch.float)
        for images, labels in test_set:
            log_ps = model(images)
            top_p, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print(f'Accuracy: {accuracy.item()*100/len(test_set)}%')


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
