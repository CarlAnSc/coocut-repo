import torch
from src.data.loaddata import load_data_func
import pytest

import os.path
@pytest.mark.skipif(not os.path.exists('data/processed/train.data'), reason="Data files not found")


def test_error_on_data_len():
    train_set, test_set = load_data_func()
    N_train = 25000
    N_test = 5000

    assert len(train_set) == N_train
    assert len(test_set) == N_test

#assert len(train_set) == N_train
#assert len(test_set) == N_test

def test_error_on_data_shape():
    train_set, test_set = load_data_func()
    shape = torch.Size([784])

    assert train_set[0][0].shape == shape
    assert test_set[0][0].shape == shape


#assert set(train_set[1].tolist()) == [0,1]

"""
assert len(dataset) == N_train for training and N_test for test
assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
assert that all labels are represented
"""