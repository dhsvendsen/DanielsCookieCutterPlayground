import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    """
    Convolutional neural network, specifically for fitting MNIST data
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward pass

            Parameters:
                    x (torch.Tensor of size [ndata, 1, 28, 28]): input images

            Returns:
                    prediction (torch.Tensor [ndata, 10]): logits of which handwritten
                    digit each image is
        """

        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 16, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 5, 5
        x = x.view(-1, 32 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        self.penult_layer = x
        x = self.fc3(x)  # -> n, 10
        return x
