from __future__ import annotations

from typing import ClassVar

from orion.testing.algo import TestPhase
from test_integration import BaseRoBOTests

from orion.algo.robo.ablr import RoBO_ABLR

N_INIT = 10


class TestRoBO_ABLR(BaseRoBOTests):
    """Tests for the ABLR algorithm."""

    algo_type: ClassVar[type[RoBO_ABLR]] = RoBO_ABLR
    config = {
        "seed": 1234,
        "n_initial_points": N_INIT,
        "maximizer": "random",
        "acquisition_func": "ei",
        "hparams": {
            "alpha": 1.0,
            "beta": 1.0,
            "learning_rate": 0.001,
            "batch_size": 10,
            "epochs": 1,
        },
        "normalize_inputs": True,
    }

    # NOTE: ABLR runtime stays much better w.r.t. number of trials than other algos in ROBO, so
    # we add a third phase with many more trials, in order to make the test suite more robust.
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("ablr", N_INIT, "suggest"),
        TestPhase("ablr_scaling", N_INIT * 5, "suggest"),
    ]
