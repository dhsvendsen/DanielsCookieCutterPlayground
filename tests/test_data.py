import numpy as np
import torch
from tests import _PATH_DATA
import pytest
import os

train_image_path = f"{_PATH_DATA}/processed/train_images.pt"
@pytest.mark.skipif(not os.path.exists(train_image_path), reason="Data files not found")
def test_len():
    train_images = torch.load(train_image_path)
    train_labels = torch.load(f"{_PATH_DATA}/processed/train_labels.pt")
    # Number
    assert len(train_images) == 40000, "Dataset did not have the correct number of samples"
    # Shape
    assert train_images.shape[1] == 28, "Wrong image height"
    assert train_images.shape[2] == 28, "Wrong image width"
    # Labels
    assert len(train_labels) == 40000, "Labels did not have the correct number of samples"
    assert np.array_equiv( np.unique(train_labels.numpy()), np.array([0,1,2,3,4,5,6,7,8,9]) ), "Not all labels are represented"
