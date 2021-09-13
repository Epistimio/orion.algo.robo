""" WIP: Working on re-implementing [Scalable HyperParameter Transfer Learning](
    https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning)

"""
import itertools
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence
import warnings
import numpy as np
import torch
import tqdm
from robo.models.base_model import BaseModel
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from warmstart import HyperParameters
from warmstart.knowledge_base import KnowledgeBase
from warmstart.tasks import Task
from warmstart.utils import zip_dicts, compute_identity
from logging import getLogger as get_logger
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from orion.algo.robo.ablr.normal import Normal
from orion.algo.robo.ablr.ablr_model import ABLR
from orion.algo.robo.ablr.registry import Registry

logger = get_logger(__name__)


def get_task_hash(task: Task) -> str:
    # TODO: Return a unique id/str 'key' for `task`.
    return task.hash
    # return hash(task)
    raise NotImplementedError(f"TODO: support task {task}")


class MultiTaskABLR(nn.Module, BaseModel):
    """ WIP: Multi-task ABLR.
    """

    def __init__(
        self,
        space_or_task: Union[Task, Dict],
        task_id: int = 0,
        registry: Registry = None,
        warm_start_data: List[Tuple[Task, Tuple[np.ndarray, np.ndarray]]] = None,
        model_factory: Callable[[Task], BaseModel] = None,
        rng: np.random.RandomState = None,
        learning_rate: float = 1e-3,
        batch_size: int = 10_000,
        epochs: int = 10,
    ):
        super().__init__()
        # self.task: Optional[Task] = None
        self.space: Dict
        self.task_id: int

        # A unique 'hash' for our task.
        # TODO: Why did we need this again?
        self.task_hash: str
        if isinstance(space_or_task, Task):
            # self.task = space_or_task
            self.space = space_or_task.full_space
            self.task_id = space_or_task.task_id
            self.task_hash = self.task.hash
        else:
            assert isinstance(space_or_task, dict)
            # TODO: Properly debug&test this case.
            from warmstart.utils.api_config import hparam_class_from_orion_space_dict
            # hparams = hparam_class_from_orion_space_dict(space_or_task)
            # task = Task(task_id)
            # TODO: Why do we actually need this 'task' again? Is it just so we have
            # something hashable?
            # self.HParams =
            self.space = space_or_task
            self.task_id = task_id
            self.task_hash = compute_identity(**self.space)

        self.rng = rng or np.random

        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size

        # TODO: domain doesn't currently work for string (categorical) inputs.
        # self.input_dims = len(self.task.get_domain().lower)

        self.registry = registry or Registry()

        # If we're not explicitly given a factory to use for creating models,
        # use the 'get_surrogate_model' method on the registry.
        if not model_factory:
            model_factory = self.registry.get_surrogate_model

        # TODO: Add a way to specify the type of feature map to be used in all
        # models.

        self.model_factory: Callable[[Task], BaseModel] = model_factory

        warm_start_data = warm_start_data or []

        # List of all known tasks.
        self.tasks: List[Task] = []
        # List of 'hashes' for each task.
        self.task_hashes: List[str] = []
        # Dict mapping from task 'hash' to task object.
        # self.task_dict: Dict[str, Task] = {}
        # Contains the points from all tasks.
        self.points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Dict mapping from a task's 'hash' to the surrogate model for that task.
        self.models: Dict[str, ABLR] = nn.ModuleDict()

        # IDEA: Use a scaling factor for the loss of each task?
        self.task_loss_coefficients: Dict[str, float] = {self.task_hash: 1}

        self.optimizer: Optimizer = None  # type: ignore

        for task, (x_t, y_t) in warm_start_data:
            logger.debug(f"Starting with {len(x_t)} points from task {task}")
            # Add the task and its data to the model.
            self.add_task(task, x_t=x_t, y_t=y_t, model=self.model_factory(task))

        # If there is no warm-start data for the current task, we still want to
        # create the surrogate model.
        if self.task_hash not in self.task_hashes:
            model = self.model_factory(self.space)
            self.add_task(self.space, x_t=[], y_t=[], model=model)
        self.optimizer: Optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def add_task(
        self, task_or_space: Union[Task, dict], x_t: np.ndarray, y_t: np.ndarray, model: BaseModel = None
    ) -> None:
        """ Adds a 'task' to the model, along with its data, and, (optionally)
        the surrogate model to use for that task.
        """
        if isinstance(task_or_space, Task):
            task = task_or_space
            space = task.space
            task_hash = task.hash
            if task_hash in self.task_hashes:
                index = self.task_hashes.index(task_hash)
                raise RuntimeError(
                    f"Task hash {task_hash} is already associated "
                    f"with task {self.tasks[index]}"
                )
        else:
            space = task_or_space
            task_hash = compute_identity(**space)
        # self.tasks.append(task)
        self.task_hashes.append(task_hash)
        # self.task_dict[task_hash] = task

        x_t = np.asarray(x_t)
        y_t = np.asarray(y_t)
        # task_input_dims = len(task.get_domain().lower)

        # if len(x_t):
        #     assert x_t.shape[-1] == task_input_dims, (x_t.shape, task_input_dims)
        self.points[task_hash] = (np.asarray(x_t), np.asarray(y_t))
        # Use the given model or create a new one.
        model = model or self.model_factory(space)

        self.models[task_hash] = model

        if self.optimizer is not None:
            ## NOTE: This wouldn't work, because some of the parameters are shared
            ## between the models, and so they are shared across 'parameter groups'
            # self.optimizer.add_param_group({"params": list(model.parameters())})
            self.update_optimizer()

    def add_data_for_task(
        self,
        task: Task,
        x_t: np.ndarray,
        y_t: np.ndarray,
        remove_duplicates: bool = True,
    ) -> None:
        """ Adds the given data to the dataset for task `task`.
        
        if `remove_duplicates` is True, removes any duplicated entries from the
        resulting dataset.
        """
        assert task in self.tasks
        index = self.tasks.index(task)
        task_hash = get_task_hash(task)
        x, y = self.points[task_hash]

        x_y_t = np.hstack([x_t, y_t.reshape([-1, 1])])
        if len(x):
            # Stack the old and new data.
            x_y = np.hstack([x, y.reshape([-1, 1])])
            new_x_y = np.concatenate([x_y, x_y_t], axis=0)
        else:
            new_x_y = x_y_t

        if remove_duplicates:
            new_x_y = np.unique(new_x_y, axis=0)

        new_x, new_y = new_x_y[:, :-1], new_x_y[:, -1:]
        self.points[task_hash] = (new_x, new_y.reshape([-1]))

    def add_context_vector_if_needed(self, task: Task, x_t: np.ndarray) -> np.ndarray:
        # task_input_dims = len(task.get_domain().lower)
        # from warmstart.tasks.quadratics import QuadraticsTaskWithContext
        # TODO: Removed the need for this, since we use tasks with context as
        # part of their space when needed.
        return x_t

    def remove_task(self, task: Task) -> None:
        """ Removes a 'task' from the model, along with its data, and, and its
        surrogate model.
        """
        task_hash = get_task_hash(task)
        self.tasks.remove(task)
        self.task_hashes.remove(task_hash)
        self.task_dict.pop(task_hash)
        self.points.pop(task_hash)
        self.models.pop(task_hash)

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update the model with the new additional data.
        
        This adds the new data from `X` and `y` to `self.points` (removing any
        duplicates) and creates surrogate models for any new tasks that don't
        already have one in `self.models`.
        
        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        # BUG: Maybe robo isn't calling update()?
        assert False, "BUG: This should be getting called, what's going on?"
        self.absorb_data(X=X, y=y)
        self.train(X=None, y=None)

    def train(self, X: np.ndarray = None, y: np.ndarray = None, do_optimize=None):
        """ Trains the model using the stored data from all tasks, as well as
        from `X` and `y`, if given.

        If `X` and `Y` are given, first calls `self.absorb_data(X, y)` to add
        the data and create any missing surrogate models for all tasks.

        NOTE: This uses the "task hashes" instead of "task ids", because it
        might be the case later on that this uses data from multiple different
        types of tasks, and SVM(task_id=1) is not the same as FCNet(task_id=1),
        so we want a way to differentiate between them to fetch the right model.
        """
        # assert (
        #     X.shape[-1] == self.input_dims
        # ), f"Data doesn't fit domain of target task {self.task}!"

        if X is not None and y is not None:
            self.absorb_data(X=X, y=y)
            X, y = None, None

        # Update the datasets on each model.
        for task_id, (model, (x_t, y_t)) in zip_dicts(
            dict(self.models.items()), self.points
        ):
            model.X = torch.as_tensor(x_t, dtype=torch.float)
            model.y = torch.as_tensor(y_t, dtype=torch.float).reshape([-1])

        # The datasets items will be [x, y, t_i], with t_i being the index where
        # the hash of the corresponding task can be found in `self.task_hashes`.
        # NOTE: Doing this mainly because I don't have much experience with
        # string tensors, so I'd rather not add the 'task hashes' in the dataset.
        datasets: Dict[str, TensorDataset] = {
            task_id: TensorDataset(
                torch.as_tensor(x_t, dtype=torch.float),
                torch.as_tensor(y_t, dtype=torch.float),
                torch.as_tensor([task_id] * len(x_t)),
            )
            for task_id, (x_t, y_t) in self.points.items()
        }

        n_samples: Dict[str, int] = {k: len(d) for k, d in datasets.items()}
        total_samples = sum(n_samples.values())

        logger.debug(f"Samples from each task: {n_samples}")
        # IDEA: Scale the losses for each task based on the number of samples?
        # IDEA: Scale using some 'distance' between the tasks?
        self.task_loss_coefficients = {
            task_hash: n_samples_for_task / total_samples
            for task_hash, n_samples_for_task in n_samples.items()
        }

        # Create a single dataset, with the data from all tasks.
        fused_dataset = ConcatDataset(datasets.values())
        # Useful for debugging:
        # dataloader = DataLoader(fused_dataset, batch_size=11, shuffle=True)
        dataloader = DataLoader(fused_dataset, batch_size=self.batch_size, shuffle=True)
        # IDEA: Cycle the shorter loaders so they all have the same length?
        all_task_hashes = np.array(self.task_hashes)

        for epoch in tqdm.tqdm(range(self.epochs), desc="Epoch", leave=False):
            inner_pbar = tqdm.tqdm(dataloader, desc="Step", leave=False)

            for x_ts, y_ts, task_labels in inner_pbar:
                self.optimizer.zero_grad()

                loss = self.get_loss(x_ts, y_ts, task_labels)
                loss.backward()

                inner_pbar.set_postfix(
                    {"Loss:": f"{loss.item():.3f}",}
                )
                # This is a bit weird, I'm not used to having to do this:
                # def closure():
                #     return self.get_loss()

                self.optimizer.step()

    def predict_dist(self, x_test: Union[Tensor, np.ndarray]) -> Normal:
        # TODO: Need to remove the 'context' portion? or not?
        # FIXME: Assumming that x_test is only ever from the target task (for now)
        x_test = self.add_context_vector_if_needed(self.task, x_test)
        self.model.X = self.X
        self.model.y = self.y

        # TODO: Debug why no points ever get registered added for the target
        # task!

        if not len(self.X):
            logger.warning(
                RuntimeWarning(
                    f"Ask to predict, but haven't observed a single point from the target task yet!"
                )
            )
            # Use the average predicted distribution from the models of each task?
            pred_distributions: Dict[str, Normal] = {}
            for t, model in self.models.items():
                if t == self.task_hash:
                    # Skip the model for the target task (as it has no data).
                    continue
                else:
                    if len(self.tasks) == 11:
                        # Debugging: Should only be 10 tasks, not 11 (1 target, 9 'other' tasks.)
                        print(f"self.task: {self.task}")
                        print("\n", *self.tasks, sep="\n")
                        raise RuntimeError(f"There's something wrong!")

                    # Set the points on the model, so it can actually give us a loss?
                    x_t, y_t = self.points[t]
                    model.X = x_t
                    model.y = y_t
                    pred_distributions[t] = model.predict_dist(x_test)

            mean_distribution = sum(pred_distributions.values()) / len(
                pred_distributions
            )
            return mean_distribution

        return self.model.predict_dist(x_test)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(X_test)
        y_pred_dist = self.predict_dist(X_test)
        return y_pred_dist.mean.numpy(), y_pred_dist.variance.numpy()

    def get_loss(self, x, y, t):
        # Returns the negative marginal log likelihood part of the result of
        # `forward`.
        return self(x, y, t)[0]

    def forward(
        self, x: Tensor, y: Tensor, task_hash_indices: Tensor = None
    ) -> Tuple[Tensor, Callable[[Tensor], Normal]]:
        """ Performs the forward pass for each task.
        
        Returns the loss, and a function to predict the distribution of y for a
        given test x.
        """

        if task_hash_indices is None:
            # The data is assumed to come from the 'target' task.
            # Get the loss and prediction for the model of the target task.
            task_hash = self.task_hash
            model = self.models[self.task_hash]
            return model(x, y)

        # Find all the tasks present in the batch. This operates on the
        # task hash indices, because they are 1-to-1 with task hashes, and of
        # integer dtype, so simpler to handle.
        all_indices = np.arange(len(x))
        all_task_hashes = np.asarray(self.task_hashes)
        unique_task_hash_indices, indices = np.unique(
            task_hash_indices, return_inverse=True
        )
        unique_task_hashes = all_task_hashes[unique_task_hash_indices]

        # Get the task hashes for each sample in the batch
        task_hashes = all_task_hashes[task_hash_indices.numpy()]

        if len(unique_task_hashes) == 1:
            # All data comes from the same task: return the predictions from the
            # right model.
            task_hash = unique_task_hashes[0]
            task = self.task_dict[task_hash]
            model = self.models[task_hash]
            return model(x, y)

        # The batch contains a mix of samples.

        # Create a 'loss' tensor for the merged loss.
        total_loss: Tensor = torch.zeros(1)
        task_indices: Dict[str, np.ndarray] = {}
        task_results: Dict[str, Any] = {}

        # Iterate over all tasks present in the batch:
        for i, task_hash in enumerate(unique_task_hashes):
            assert (
                task_hash in self.task_hashes
            ), f"Weird, no task with hash {task_hash}"

            # task: Task = self.task_dict[task_hash]
            model = self.models[task_hash]
            # Create an array to select all entries from this task in the batch.
            task_indices_in_batch = all_indices[indices == i]
            task_indices[task_hash] = task_indices_in_batch

            x_t = x[task_indices_in_batch]
            y_t = y[task_indices_in_batch]

            # Get the loss and predictive function from each model.
            task_result = model(x_t, y_t)
            task_results[task_hash] = task_result

        return self._merge_results(task_indices, task_results)

    def _merge_results(
        self,
        task_indices: Dict[str, Sequence[int]],
        task_results: Dict[str, Tuple[Tensor, Callable]],
    ) -> Tuple[Tensor, Callable]:
        """ Merges the results from the forward pass of the models for each task. """
        total_loss: Tensor = torch.zeros(1)
        pred_fns: List[Callable] = []

        def _pred_fn(x_test: np.ndarray):
            raise NotImplementedError(
                "TODO: a function which redirects to the "
                "right pred_fn from the task_results!"
            )

        # TODO: Should we scale down some of the losses, depending
        # on the size of the datasets?

        for task_label, (task_loss, task_pred_fn) in task_results.items():
            logger.debug(f"Loss for task {task_label}: {task_loss}")
            task_loss_coefficient = self.task_loss_coefficients[task_label]
            total_loss += task_loss_coefficient * task_loss

        return total_loss, _pred_fn

    def absorb_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Adds data from one or more tasks to the model, constructing the
        surrogate models for any new tasks.

        This updates `self.points`, `self.models`, etc.
        """
        assert self.task in self.tasks
        assert self.task_hash in self.task_hashes
        n_samples = len(X)
        # TODO: This only works if all tasks have the same number of input
        # dimensions atm.

        # TODO: Since the task now has the context as part of its space, We'll
        # have to assume that all the points are from the target task for now,
        # and that train(X, y) only ever gets called with data from the target
        # task.
        # Before, the goal was to figure out which task the data belongs
        # to by matching the context vector portion, but now that it's in the
        # space, the acquisition function proposes points that have a different
        # value from a2, a1, a0, which don't match any of the existing tasks.
        assert False, (X.shape, X[0])
        input_dims = len(self.task.get_domain().lower)
        assert len(X.shape) == 2
        assert X in self.task.get_domain()
        self.add_data_for_task(self.task, X, y)
        return

        task_info = X[..., input_dims:]
        x = X[..., :input_dims]
        assert len(task_info.shape) == 2

        unique_task_ids, indices = np.unique(task_info, axis=0, return_inverse=True)
        all_indices = np.arange(n_samples)
        # Split up the data into groups for each unique 'task id'.
        for i, unique_task_id in enumerate(unique_task_ids):

            task_indices = all_indices[indices == i]
            # mask = all_indices[]
            x_t = x[task_indices]
            y_t = y[task_indices]

            if unique_task_id.shape[-1] == 1:
                task_id = unique_task_id.item()
                context_vector = None
            else:
                task_id = None
                context_vector = unique_task_id

            existing_task: Optional[Task] = None
            for task in self.tasks:
                if (task_id is not None and task.task_id == task_id) or (
                    context_vector is not None
                    and np.allclose(task.context_vector, context_vector, 1e-4)
                ):
                    existing_task = task
                    break

            if existing_task:
                self.add_data_for_task(task, x_t, y_t)
            else:
                # Either there are currently no registered tasks, or no task
                # matched the given task ID and/or "context" vector.
                assert context_vector is not None
                # TODO: Fix this: if we don't find an existing task, we
                # create it (assuming that it's a QuadraticsTask.)
                # We could also maybe use a Knowledge base to test the points
                # against the spaces of existing tasks ?
                domain = self.task.get_domain()
                assert x in domain, "Assuming points are from self.task"
                task_id = max([task.task_id for task in self.tasks], default=0)
                new_task = type(self.task)(
                    *context_vector, task_id=task_id, rng=self.task.rng
                )

                self.add_task(new_task, x_t, y_t)

    def update_optimizer(self) -> None:
        state_dict = self.optimizer.state_dict()
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        # TODO: Actually reuse the optimizer state (here we have to dump it,
        # because the parameter groups dont match?
        try:
            self.optimizer.load_state_dict(state_dict)
        except ValueError as e:
            warnings.warn(
                RuntimeWarning(
                    f"Re-creating optimizer (unable to load the previous state: {e})"
                )
            )

    @property
    def model(self) -> ABLR:
        return self.models[self.task_hash]

    @property
    def n_samples(self) -> Dict[str, int]:
        return {k: len(x) for k, (x, _) in self.points.items()}

    @property
    def X(self) -> np.ndarray:
        return self.points[self.task_hash][0]

    @X.setter
    def X(self, value: np.ndarray) -> None:
        _, y = self.points[self.task_hash]
        self.points[self.task_hash] = (value, y)

    @property
    def y(self) -> np.ndarray:
        return self.points[self.task_hash][1]

    @y.setter
    def y(self, value: np.ndarray) -> None:
        x, _ = self.points[self.task_hash]
        self.points[self.task_hash] = (x, value)

    def __repr__(self):
        """ 'patch' for the inifine recursion that happens in repr when using a
        Registry and the models ModuleDict.
        """
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            from torch.nn.modules.module import _addindent

            if key == "registry" or key == "models":
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        # TODO: Not sure why this was ever implemented!
        if self.n_samples[self.task_hash] == 0:
            # Weird, haven't observed a single 'real' point yet, but we're being
            # asked what minimum to give..
            other_points = {
                t: (x_t, y_t)
                for t, (x_t, y_t) in self.points.items()
                if t != self.task_hash
            }
            best_indices = {t: np.argmin(y_t) for t, (x_t, y_t) in other_points.items()}
            best_from_other_tasks = {
                t: (x_t[best_index], y_t[best_index])
                for t, ((x_t, y_t), best_index) in zip_dicts(other_points, best_indices)
            }
            # Take the best point from the tasks of the same type as the current
            # task.
            best_x: np.ndarray
            best_y: float
            task: Task
            for t, (best_x, best_y) in sorted(
                best_from_other_tasks.items(), key=lambda tup: tup[1][1]
            ):
                task = self.task_dict[t]
                if isinstance(task, type(self.task)):
                    logger.debug(
                        f"Taking the best point from task {task} as a "
                        f"substitute, since there aren't any points from the "
                        f"target task ({self.task})."
                    )
                    break
            else:
                raise NotImplementedError("Found no compatible task?!")
        else:
            best_index = np.argmin(self.y)
            best_x = self.X[best_index]
            best_y = self.y[best_index]
            task = self.task

        best_x = np.atleast_1d(best_x)
        best_y = np.atleast_1d(best_y)
        # if self.add_context:
        #     best_x = self.add_context_vector_if_needed(self.task, best_x)
        # assert False, best_x.shape
        return best_x, best_y
