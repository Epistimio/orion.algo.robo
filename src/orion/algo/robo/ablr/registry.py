from typing import Dict, Type, Union

import torch
from torch import nn, Tensor
from simple_parsing.helpers.hparams import HyperParameters
from .utils import compute_identity
from logging import getLogger as get_logger
from orion.benchmark.task.base import BaseTask
from blitz.modules import BayesianLinear

# from .dngo_reimplementation import BayesianLinear
from orion.algo.robo.ablr.encoders import Encoder, NeuralNetEncoder, AdaptiveEncoder
from orion.algo.robo.ablr.ablr_model import ABLR

logger = get_logger(__name__)


class Registry(nn.Module):
    """ Holds the components (feature maps and Bayesian regression params alphas
    and betas) for each task.
    
    Used to create the models for each task so as to have the feature maps be
    shared between tasks, while each task gets its own alpha and beta
    parameters.
    """

    def __init__(self):
        super().__init__()
        # Dict of feature maps for each search space.
        self.feature_nets: Dict[str, Encoder] = nn.ModuleDict()
        # Dict of bayesian layers for each search space + task_id.
        # (i.e. the feature nets are shared for different tasks, but not the
        # bayesian layers)
        self.bayesian_regressor_layers: Dict[str, BayesianLinear] = nn.ModuleDict()

        # Stores the constructed objects, so we don't create a new regressor
        # every time.
        self.surrogate_models: Dict[str, ABLR] = {}

        self.known_hparam_codes: Dict[str, nn.Parameter] = nn.ParameterDict()

    def get_surrogate_model_for(self, task: BaseTask, feature_dims: int = 50) -> ABLR:
        # TODO: This could be an instance method on the ProfetTask class!
        space = task.hparams.get_orion_space_dict()
        task_id = task.task_id
        print(f"Getting surrogate model for task id {task_id}.")
        print(f"(# of existing surrogate models: {len(self.surrogate_models)})")
        return self.get_surrogate_model(
            space, task_id=task_id, feature_dims=feature_dims
        )

    def get_surrogate_model(
        self,
        space: Union[Dict, Type[HyperParameters], BaseTask],
        task_id: Union[int, str] = 0,
        feature_dims: int = 50,
        encoder_type: Type[Encoder] = NeuralNetEncoder,
    ) -> ABLR:
        """ Gets or creates a surrogate model for the given space, task_id and
        number of feature dimensions.
        
        The feature maps are shared between different values of task_id for
        equivalent search spaces, but each task id in a given space has its own
        bayesian linear regression layer.
        
        This is equivalent to indexing into a dictionary with keys like so:
        feature_maps: <space_id, feature_dims>
        bayesian_regressors: <space_id, feature_dims, task_id>

        Parameters
        ----------
        space : Union[Dict, Type[HyperParameters]]
            Either an orion-formatted 'space' dict, or a HyperParameters object or class.
        task_id : Union[int, str], optional
            An unique id for this task, for example the dataset name. Defaults
            to 0.
        feature_dims : int, optional
            Dimensionality of the feature space, by default 50.

        Returns
        -------
        ABLR
            A new (or existing) surrogate model.
        """
        if isinstance(space, BaseTask):
            task_id = space.task_id
            space = space.full_space
        space = self.to_space_dict(space)
        space_id = self.get_space_id(space)

        feature_map_id = space_id
        bayesian_layer_id = f"{space_id}_{task_id}"
        surrogate_model_id = bayesian_layer_id

        if surrogate_model_id in self.surrogate_models:
            # We already created a surrogate model for this space/task_id.
            return self.surrogate_models[surrogate_model_id]

        feature_map = self.get_feature_map(
            space, feature_dims=feature_dims, encoder_type=encoder_type
        )
        bayesian_regressor = self.get_regression_layer(space, feature_dims=feature_dims)
        # Save the constructed regressor so we don't re-create it from scractch
        # next time.
        surrogate_model = ABLR(space, feature_map=feature_map,)
        self.surrogate_models[surrogate_model_id] = surrogate_model
        return surrogate_model

    def get_hparam_code(
        self, hparam: str, hparam_space: str, feature_dims: int = 50
    ) -> nn.Parameter:
        """Get the encoding vector / bag-of-words code for the given hparam.

        Parameters
        ----------
        hparam : str
            The Hyper-parameter name.
        hparam_space : str
            The hyper-parameter space for that hyper-parameter, as an
            orion-formatted string, e.g. 'uniform(0, 1)'.
        feature_dims : int, optional
            The feature dimensions, by default 50

        Returns
        -------
        nn.Parameter
            a nn.Parameter containing a FloatTensor of shape [feature_dims, 1]
            that can be used as part of the first linear layer of a neural net
            encoder, for instance.
        """
        # Create a unique 'identifier' for this hparam.
        # TODO: We could maybe do something a bit fancier here?
        key = f"{hparam}_{hparam_space}_{feature_dims}"
        key = key.replace(".", ",")
        if key in self.known_hparam_codes:
            logger.debug(f"Found existing code for hparam {hparam} at key {key}")
            return self.known_hparam_codes[key]
        code: Tensor = torch.randn([feature_dims, 1])
        code = torch.nn.init.xavier_normal_(code)
        code = nn.Parameter(code)
        self.known_hparam_codes[key] = code
        return code

    def get_feature_map(
        self,
        space: Union[Dict, Type[HyperParameters]],
        feature_dims: int = 50,
        use_shared_embeddings: bool = True,
        encoder_type: Type[Encoder] = NeuralNetEncoder,
    ) -> Encoder:
        """ Get or create the shared feature map model for this space and 
        number of feature dimensions.
        
        TODO: Maybe add a parameter to control wether we only provide an
        initialization, or if we actually give back a 'clone' (i.e. if we allow
        the task to change the values of this shared Parameter).
        """
        space_dict = self.to_space_dict(space)
        space_id = self.get_space_id(space_dict)
        # Assuming that each entry in the space means one float value.

        feature_map_id = f"{encoder_type.__name__}_{space_id}_o{feature_dims}"
        if feature_map_id not in self.feature_nets:
            # Create a new 'feature map' if needed.
            # Testing this out:
            if use_shared_embeddings:
                encoder = AdaptiveEncoder(
                    input_space=space_dict,
                    out_features=feature_dims,
                    registry=self,  # TODO: Don't pass this 'self'! Causes lots of trouble!
                )
            else:
                encoder = encoder_type(
                    input_space=space_dict, out_features=feature_dims,
                )
            self.feature_nets[feature_map_id] = encoder
        else:
            encoder = self.feature_nets[feature_map_id]

        return encoder

    def get_regression_layer(
        self, space: Union[Dict, Type[HyperParameters]], feature_dims: int = 50
    ) -> BayesianLinear:
        """ Get or create the bayesian regression layer for this space and 
        number of feature dimensions.
        """
        space = self.to_space_dict(space)
        space_id = self.get_space_id(space)

        bayesian_layer_id = f"{space_id}_o{feature_dims}"
        if bayesian_layer_id not in self.bayesian_regressor_layers:
            # Create a new Bayesian Linear Regressor layer if needed.
            self.bayesian_regressor_layers[bayesian_layer_id] = BayesianLinear(
                in_features=feature_dims, out_features=1,
            )
        bayesian_regressor = self.bayesian_regressor_layers[bayesian_layer_id]
        return bayesian_regressor

    @staticmethod
    def to_space_dict(
        space: Union[Dict, HyperParameters, Type[HyperParameters]]
    ) -> Dict:
        if isinstance(space, dict):
            return space
        if isinstance(space, HyperParameters) or issubclass(space, HyperParameters):
            return space.get_orion_space_dict()
        raise RuntimeError(f"Don't know how to get space dict of {space}")

    def get_space_id(self, space: Union[Dict, Type[HyperParameters]]) -> str:
        """ Gets the unique 'identifier' associated with this space. """
        space = self.to_space_dict(space)
        space_id = compute_identity(**space)
        return space_id

    # TODO: Add a load_state_dict method that re-creates the layers.
