import torch
from draugr.torch_utilities import cross_validation_generator, to_tensor
from torch.utils.data import TensorDataset


def asdasidoj():
    """ """
    X = to_tensor([torch.diag(torch.arange(i, i + 2)) for i in range(200)])
    x_train = TensorDataset(X[:100])
    x_val = TensorDataset(X[100:])

    for train, val in cross_validation_generator(x_train, x_val):
        print(len(train), len(val))
        print(train[0], val[0])


asdasidoj()
