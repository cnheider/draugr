__all__ = ["NamedTensorTuple"]


class NamedTensorTuple:
    """
    Help class for manage boxes, labels, etc...
    Not inherit dict due to `default_collate` will change dict's subclass to dict."""

    def __init__(self, **kwargs):
        self._data_dict = kwargs

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data_dict[key]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def _call(self, name, *args, **kwargs):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, name):
                self._data_dict[key] = getattr(value, name)(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        """

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:"""
        return self._call("to", *args, **kwargs)

    def numpy(self):
        """

        :return:
        :rtype:"""
        return self._call("numpy")

    def __repr__(self):
        return self._data_dict.__repr__()
