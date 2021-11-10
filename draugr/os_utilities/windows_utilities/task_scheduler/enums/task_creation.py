from enum import Enum


class TaskCreationEnum(Enum):
    """ """

    TASK_VALIDATE_ONLY = 1
    TASK_CREATE = 2
    TASK_UPDATE = 4
    TASK_CREATE_OR_UPDATE = 6  # If task already exists, it will be updated
    TASK_DISABLE = 8
    TASK_DONT_ADD_PRINCIPAL_ACE = 10
    TASK_IGNORE_REGISTRATION_TRIGGERS = 20
