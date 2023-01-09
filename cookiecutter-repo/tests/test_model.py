from src.models.model import MyAwesomeModel
from src.data.loaddata import load_data_func
import torch
import pytest


#model = MyAwesomeModel()
#train_set, _ = load_data_func()

#loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
#image, _ = loader

#image, label = train_set[0]
#print(image.shape)

#output, _ = model(image.unsqueeze(0))

#assert image.shape == torch.Size([784])

def test_error_on_input_shape():
    model = MyAwesomeModel()
    train_set, _ = load_data_func()
    image, label = train_set[0]
    output, _ = model(image.unsqueeze(0))

    assert image.shape == torch.Size([784]), 'Output is of wrooong shape, man'

def test_error_on_output_shape():
    model = MyAwesomeModel()
    train_set, _ = load_data_func()
    image, label = train_set[0]
    print('Hoiansræoknsræoknsaæronisronakvåosinrvåoinrsvåoinsrivnorvnoirnsv',image.shape)
    output, _ = model(image.unsqueeze(0))

    assert output.shape == torch.Size([1,10])

# Check Assert is working:
# tests/test_model.py

"""
def test_error_on_wrong_shape():
    #output, _ = model(image.unsqueeze(0))

    with pytest.raises(ValueError, match='Expected the input dimensions to be a torchTensor of size 784'):
        model = MyAwesomeModel()
        model(torch.randn(1,2,3))
"""
