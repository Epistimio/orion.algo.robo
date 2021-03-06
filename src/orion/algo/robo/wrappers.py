"""
:mod:`orion.algo.robo.wrappers -- Wrappers for RoBO Optimizers
==============================================================

.. module:: robo
    :platform: Unix
    :synopsis: Wrappers for RoBO Optimizers

Wraps RoBO optimizers to provide a uniform interface across them. Namely,
it adds the properties `lower` and `upper` and the methods `seed()`, `state_dict()`
and `set_state()`.
"""
import numpy
import torch
from pybnn.bohamiann import Bohamiann
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.wrapper_bohamiann import WrapperBohamiann, get_default_network


class OrionGaussianProcessWrapper(GaussianProcess):
    """Wrapper for GaussianProcess"""

    # pylint:disable=attribute-defined-outside-init
    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        self.rng.set_state(state_dict["model_rng_state"])
        self.prior.rng.set_state(state_dict["prior_rng_state"])
        self.kernel.set_parameter_vector(state_dict["model_kernel_parameter_vector"])
        self.noise = state_dict["noise"]

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        return {
            "prior_rng_state": self.prior.rng.get_state(),
            "model_rng_state": self.rng.get_state(),
            "model_kernel_parameter_vector": self.kernel.get_parameter_vector().tolist(),
            "noise": self.noise,
        }

    def seed(self, seed):
        """Seed all internal RNGs"""
        seeds = numpy.random.RandomState(seed).randint(1, 10e8, size=2)
        self.rng.seed(seeds[0])
        self.prior.rng.seed(seeds[1])


class OrionGaussianProcessMCMCWrapper(GaussianProcessMCMC):
    """Wrapper for GaussianProcess with MCMC"""

    # pylint:disable=attribute-defined-outside-init
    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        self.rng.set_state(state_dict["model_rng_state"])
        self.prior.rng.set_state(state_dict["prior_rng_state"])

        if state_dict.get("model_p0", None) is not None:
            # pylint:disable=invalid-name
            self.p0 = numpy.array(state_dict["model_p0"])
            self.burned = True
        elif hasattr(self, "p0"):
            delattr(self, "p0")
            self.burned = False

    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        s_dict = {
            "prior_rng_state": self.prior.rng.get_state(),
            "model_rng_state": self.rng.get_state(),
        }

        if hasattr(self, "p0"):
            s_dict["model_p0"] = self.p0.tolist()

        return s_dict

    def seed(self, seed):
        """Seed all internal RNGs"""
        seeds = numpy.random.RandomState(seed).randint(1, 10e8, size=2)
        self.rng.seed(seeds[0])
        self.prior.rng.seed(seeds[1])


class OrionBohamiannWrapper(WrapperBohamiann):
    """Wrapper for Bohamiann"""

    def __init__(
        self,
        lower,
        upper,
        learning_rate=1e-2,
        verbose=False,
        use_double_precision=True,
        **kwargs
    ):

        self.lr = learning_rate  # pylint:disable=invalid-name
        self.verbose = verbose
        self.bnn = Bohamiann(
            get_network=get_default_network,
            use_double_precision=use_double_precision,
            **kwargs
        )

        self.lower = lower
        self.upper = upper

    # pylint:disable=no-self-use
    def set_state(self, state_dict):
        """Restore the state of the optimizer"""
        torch.random.set_rng_state(state_dict["torch"])

    # pylint:disable=no-self-use
    def state_dict(self):
        """Return the current state of the optimizer so that it can be restored"""
        return {"torch": torch.random.get_rng_state()}

    def seed(self, seed):
        """Seed all internal RNGs"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        torch.manual_seed(seed)
