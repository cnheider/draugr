import os
from enum import Enum
from typing import Iterable, Union

__all__ = ["DeviceOrderEnum", "set_cuda_device_order", "set_cuda_visible_devices"]


class DeviceOrderEnum(Enum):
    fastest_first = "FASTEST_FIRST"
    pci_bus_id = "PCI_BUS_ID"


def set_cuda_device_order(order: DeviceOrderEnum = DeviceOrderEnum.pci_bus_id) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = DeviceOrderEnum(order).value


def set_cuda_visible_devices(devices: Union[str, int, Iterable[int]]) -> None:
    if isinstance(devices, int):
        devices = str(devices)
    elif isinstance(devices, Iterable):
        devices = ",".join(str(d) for d in devices)
    if devices is None:  # TODO: Nix specific, choose the least utilised device
        devices = "$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
