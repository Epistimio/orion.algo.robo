import pytest
from pathlib import Path

import torch

from orion.algo.robo.ablr.registry import Registry

@pytest.mark.skip(reason="Need to refactor this test so it doesn't depend on the profet tasks.")
def test_sharing_of_modules():
    from orion.benchmark.task.profet import FcNetTask

    # TODO: (wip) test that the feature extractor modules are shared between
    # tasks but not the bayesian layers on top.
    data_dir = Path("profet_data/data")
    task_a = FcNetTask(input_dir=data_dir)

    hps = task_a.sample(10)
    perfs = task_a(hps)

    x = torch.as_tensor([hp.to_array() for hp in hps], dtype=torch.float32)
    y = torch.as_tensor(perfs, dtype=torch.float32)

    registry = Registry()
    regressor = registry.get_surrogate_model(hps[0])
    regressor2 = registry.get_surrogate_model(type(hps[1]))
    assert regressor2 is regressor

    regressor2 = registry.get_surrogate_model(type(hps[1]).get_orion_space_dict())
    assert regressor2 is regressor

    # TODO: Spawn a different FcNet task (which might take a while if the profet
    # task with that task_idx doesnt already exist).

    # task_b = FcNetTask(data_dir, task_idx=1)
    # regressor2 = registry.get_surrogate_model(task_b.hparams)
