import numpy as np
import torch
from tests import _PATH_DATA
from src.models.model import ConvNet

# In the detox project we use a pytorch ligthning module, which we can use 
# to do call the loss directly from the model 
def test_loss():
    assert True