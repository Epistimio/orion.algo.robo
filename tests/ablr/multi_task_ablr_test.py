from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from warmstart.tasks import Task

from orion.algo.robo.ablr.multi_task_ablr import MultiTaskABLR


def warmstart_improvement(
    tasks: List[Task],
    n_samples_from_target_task: int,
    n_warmstart_samples_per_task: int = 0,
    n_epochs: int = 100,
) -> Tuple[float]:
    """ Returns the improvements from with vs without warm-starting, using a
    leave-one-task-out procedure.
    
    returns a tuple with <Mean Improvement %, Improvements variance %>
    """
    improvement_percentages: List[float] = []

    for i, target_task in enumerate(tasks):
        other_tasks = tasks[:i] + tasks[i + 1 :]

        model = MultiTaskABLR(target_task)

        target_samples = n_samples_from_target_task

        if n_warmstart_samples_per_task and not target_samples:
            # Need at least 1 point, can't start completely from scratch just yet.
            target_samples = 1
        # Make sure we sample enough points so that we can get enough test points.
        x, y = target_task.make_dataset_np(max(target_samples * 2, 1000))

        test_x, test_y = x[target_samples:], y[target_samples:]
        assert len(test_x) > 100

        if target_samples > 0:
            train_x, train_y = x[:target_samples], y[:target_samples]
            # Train on the first task:
            model.train(train_x, train_y)

        def get_test_error(
            model: MultiTaskABLR, test_x: np.ndarray, test_y: np.ndarray
        ) -> float:
            y_preds_mean, y_preds_var = model.predict(test_x)
            y_preds = y_preds_mean.reshape(-1)

            test_error = 0
            for i, (x, y_pred, y) in enumerate(zip(test_x, y_preds, test_y)):
                error = (y_pred - y) ** 2
                test_error += error
                # print(f"x: {x}, y_pred: {y_pred:.2f}, y_true: {y:.2f}, mse: {error}")
            return test_error

        test_error_before = get_test_error(model, test_x, test_y)
        print(f"Total test error (no warm starting) {test_error_before}")

        # Second example: warm starting from previous tasks:

        model = MultiTaskABLR(target_task, epochs=n_epochs)

        # Absorb data from other tasks:
        for task in other_tasks:
            task_x, task_y = task.make_dataset_np(n_warmstart_samples_per_task)
            model.add_task(task, task_x, task_y)

        # Train on the target task:
        if target_samples > 0:
            model.train(train_x, train_y)

        test_error_after = get_test_error(model, test_x, test_y)
        print(f"Total test error (with warm-starting): {test_error_after}")

        improvement = test_error_before - test_error_after
        improvement_pct = improvement / test_error_before
        print(
            f"Warm start helped: {improvement > 0} ({improvement_pct:.2%} change in test error)."
        )
        improvement_percentages.append(improvement_pct)

    improvement_mean = np.mean(improvement_percentages)
    improvement_var = np.var(improvement_percentages)
    print(f"Improvement: {improvement_mean*100:.2}+={improvement_var:.2%}")
    return improvement_mean, improvement_var


input_path: Path = Path("profet_data/data")


@pytest.mark.skip(reason="Really slow, since it requires re-training the meta-model")
def test_profet_svm():
    from warmstart.tasks.profet import SvmTask

    tasks = [SvmTask(task_id=i) for i in range(10)]
    mean_improvement, improvement_variance = warmstart_improvement(
        tasks, n_samples_from_target_task=5, n_warmstart_samples_per_task=10
    )
    assert mean_improvement >= 0.30
    assert improvement_variance > 1


@pytest.mark.xfail(
    reason="TODO: Fix this test, currently encountering negative variance"
)
def test_quadratics():
    from warmstart.tasks.quadratics import QuadraticsTask

    tasks = [QuadraticsTask(task_id=i, seed=i) for i in range(10)]
    mean_improvement, improvement_variance = warmstart_improvement(
        tasks, n_samples_from_target_task=5, n_warmstart_samples_per_task=10
    )
    assert mean_improvement >= 0.30
    assert improvement_variance > 1


@pytest.mark.xfail(reason="TODO: Fix this test, issues with categorical inputs")
def test_svm_openml():
    from warmstart.tasks.openml.svm import SvmTask

    tasks = [SvmTask(task_id=i, seed=i) for i in range(10)]
    mean_improvement, improvement_variance = warmstart_improvement(
        tasks, n_samples_from_target_task=5, n_warmstart_samples_per_task=10
    )
    assert mean_improvement >= 0.30
    assert improvement_variance > 1


if __name__ == "__main__":
    test_profet_svm()
