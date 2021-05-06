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


class TrainingScalars(Enum):
    training_loss = "training_loss"
    validation_loss = "validation_loss"
    validation_accuracy = "validation_accuracy"
    new_best_model = "new_best_model"


class TrainingCurves(Enum):
    validation_precision_recall = "validation_precision_recall"
    validation_receiver_operator_characteristic = (
        "validation_receiver_operator_characteristic"
    )


class TrainingTables(Enum):
    validation_confusion_matrix = "validation_confusion_matrix"
    validation_support = "validation_support"


class TestingScalars(Enum):
    test_accuracy = "test_accuracy"
    test_precision = "test_precision"
    test_recall = "test_recall"
    test_receiver_operator_characteristic_auc = (
        "test_receiver_operator_characteristic_auc"
    )


class TestingCurves(Enum):
    test_precision_recall = "test_precision_recall"
    test_receiver_operator_characteristic = "test_receiver_operator_characteristic"


class TestingTables(Enum):
    test_confusion_matrix = "test_confusion_matrix"
    test_support = "test_support"


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
