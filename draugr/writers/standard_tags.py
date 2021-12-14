from enum import Enum

__all__ = [
    "TrainingScalars",
    "TrainingCurves",
    "TrainingTables",
    "TestingScalars",
    "TestingCurves",
    "TestingTables",
    "should_plot_y_log_scale",
    "should_smooth_series",
]

from sorcery import assigned_names


class TrainingScalars(Enum):
    """ """

    (
        training_loss,
        validation_loss,
        validation_accuracy,
        new_best_model,
    ) = assigned_names()


class TrainingCurves(Enum):
    """ """

    (
        validation_precision_recall,
        validation_receiver_operator_characteristic,
    ) = assigned_names()


class TrainingTables(Enum):
    """ """

    validation_confusion_matrix, validation_support = assigned_names()


class TestingScalars(Enum):
    """ """

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_receiver_operator_characteristic_auc,
    ) = assigned_names()


class TestingCurves(Enum):
    """ """

    test_precision_recall, test_receiver_operator_characteristic = assigned_names()


class TestingTables(Enum):
    """ """

    test_confusion_matrix, test_support = assigned_names()


def should_plot_y_log_scale(tag: Enum) -> bool:
    """

    :param tag:
    :return:
    """
    if tag is TrainingScalars.training_loss:
        return True
    elif tag is TrainingScalars.validation_loss:
        return True
    return False


def should_smooth_series(tag: Enum) -> bool:
    """

    :param tag:
    :return:
    """
    if tag is TrainingScalars.training_loss:
        return True
    elif tag is TrainingScalars.validation_loss:
        return True
    return False


if __name__ == "__main__":
    print(TrainingScalars.training_loss.value)
