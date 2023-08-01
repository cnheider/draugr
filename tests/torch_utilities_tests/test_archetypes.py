import numpy
import pytest
import torch

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)

from draugr.torch_utilities import get_num_samples, get_archetypes


@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_get_num_samples(dtype):
    """

    :param dtype:
    :type dtype:
    :return:
    :rtype:
    """
    # Numpy
    num_classes = 3
    targets_np = numpy.random.randint(0, num_classes, size=(2, 5))

    # PyTorch
    targets_th = torch.as_tensor(targets_np)
    num_samples_th = get_num_samples(targets_th, num_classes, dtype=dtype)

    num_samples_np = numpy.zeros((2, num_classes), dtype=numpy.int_)
    for i in range(2):
        for j in range(5):
            num_samples_np[i, targets_np[i, j]] += 1

    assert num_samples_th.shape == (2, num_classes)
    if dtype is not None:
        assert num_samples_th.dtype == dtype
    numpy.testing.assert_equal(num_samples_th.numpy(), num_samples_np)


def test_get_archetypes():
    """

    :return:
    :rtype:
    """
    # Numpy
    num_classes = 3
    embeddings_np = numpy.random.rand(2, 5, 7).astype(numpy.float32)
    targets_np = numpy.random.randint(0, num_classes, size=(2, 5))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    targets_th = torch.as_tensor(targets_np)
    archetypes_th = get_archetypes(embeddings_th, targets_th, num_classes)

    assert archetypes_th.shape == (2, num_classes, 7)
    assert archetypes_th.dtype == embeddings_th.dtype

    archetypes_np = numpy.zeros((2, num_classes, 7), dtype=numpy.float32)
    num_samples_np = numpy.zeros((2, num_classes), dtype=numpy.int_)
    for i in range(2):
        for j in range(5):
            num_samples_np[i, targets_np[i, j]] += 1
            for k in range(7):
                archetypes_np[i, targets_np[i, j], k] += embeddings_np[i, j, k]

    for i in range(2):
        for j in range(num_classes):
            for k in range(7):
                archetypes_np[i, j, k] /= max(num_samples_np[i, j], 1)

    numpy.testing.assert_allclose(archetypes_th.detach().numpy(), archetypes_np)
