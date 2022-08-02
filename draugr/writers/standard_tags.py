from enum import Enum

__all__ = [
    "StandardTrainingScalarsEnum",
    "StandardTrainingCurvesEnum",
    "StandardTrainingTablesEnum",
    "StandardTestingScalarsEnum",
    "StandardTestingCurvesEnum",
    "StandardTestingTablesEnum",
    "should_plot_y_log_scale",
    "should_smooth_series",
]

from sorcery import assigned_names


class StandardTrainingScalarsEnum(Enum):
    """description"""

    (
        training_loss,
        validation_loss,
        validation_accuracy,
        new_best_model,
    ) = assigned_names()


class StandardTrainingCurvesEnum(Enum):
    """description"""

    (
        validation_precision_recall,
        validation_receiver_operator_characteristic,
    ) = assigned_names()


class StandardTrainingTablesEnum(Enum):
    """description"""

    validation_confusion_matrix, validation_support = assigned_names()


class StandardTestingScalarsEnum(Enum):
    """description"""

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_receiver_operator_characteristic_auc,
    ) = assigned_names()


class StandardTestingCurvesEnum(Enum):
    """description"""

    test_precision_recall, test_receiver_operator_characteristic = assigned_names()


class StandardTestingTablesEnum(Enum):
    """description"""

    test_confusion_matrix, test_support = assigned_names()


def should_plot_y_log_scale(tag: Enum) -> bool:
    """

    :param tag:
    :return:
    """
    if tag is StandardTrainingScalarsEnum.training_loss:
        return True
    elif tag is StandardTrainingScalarsEnum.validation_loss:
        return True
    return False


def should_smooth_series(tag: Enum) -> bool:
    """

    :param tag:
    :return:
    """
    if tag is StandardTrainingScalarsEnum.training_loss:
        return True
    elif tag is StandardTrainingScalarsEnum.validation_loss:
        return True
    return False


if __name__ == "__main__":
    print(StandardTrainingScalarsEnum.training_loss.value)
