from sorcery import assigned_names


def upper_case_assigned_names():
    """

    :return:
    :rtype:
    """
    return {name.upper(): value for name, value in assigned_names().items()}


def lower_case_assigned_names():
    """

    :return:
    :rtype: object
    """
    return {name.lower(): value for name, value in assigned_names().items()}
