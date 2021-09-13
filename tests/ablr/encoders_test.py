import copy
import pytest

from orion.algo.robo.ablr.registry import Registry
from orion.algo.robo.ablr.encoders import AdaptiveEncoder


@pytest.fixture()
def registry():
    registry = Registry()
    yield registry


def test_shared_hparam_code(registry: Registry):
    """ Test that when using the 'AdaptiveEncoder' (name might have changed),
    when backpropagating through that encoder, the grads are also reflected in
    the registry's version of that parameter.
    """

    input_space = {
        "learning_rate": "log_uniform(1e-9, 1e-1)",
        "dropout_prob": "uniform(0, 0.8)",
    }
    # TODO


def test_deepcopy_of_adaptive_encoder():
    model = AdaptiveEncoder({"foo": "uniform(0,1)"}, out_features=10)
    copy.deepcopy(model)
