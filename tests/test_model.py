import pytest
import torch

from src.models.model import ConvNet
from tests import _PATH_DATA

n = 100
train_images = torch.unsqueeze(
    torch.load(f"{_PATH_DATA}/processed/train_images.pt")[0:n], dim=1
)
model = ConvNet()
output = model(train_images)


@pytest.mark.parametrize("test_input,expected", [(10, 10), (20, 20), (42, 42)])
def test_output(test_input, expected):
    assert model(train_images[0:test_input]).shape[0] == expected


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
