from typing import Callable, Tuple, TypeVar

import jax
from jax import numpy as jnp
from jax.experimental.pjit import with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec

import haliax as hax
from haliax import Axis, auto_sharded
from haliax.jax_utils import named_call
from haliax.partitioning import ResourceAxis, ResourceMapping, shard_with_axis_mapping
from haliax.util import is_named_array
from levanter.jax_utils import reduce


M = TypeVar("M")
X = TypeVar("X")


@named_call
def accumulate_gradients(f: Callable[[M, X], Tuple[float, M]], model: M, *inputs: X) -> Tuple[float, M]:
    zero = (jnp.zeros(()), jax.tree_util.tree_map(lambda m: jnp.zeros_like(m), model), 0)

    def compute_and_accumulate(acc, *input):
        loss, grad = f(model, *input)
        acc_loss, acc_grad, n = acc
        return loss + acc_loss, jax.tree_map(jnp.add, acc_grad, grad), n + 1

    total_loss, total_grad, total_n = reduce(compute_and_accumulate, zero, *inputs)

    return total_loss / total_n, jax.tree_map(lambda x: x / total_n, total_grad)


# cf https://github.com/google-research/t5x/blob/main/t5x/trainer.py#L617
@named_call
def accumulate_gradients_sharded(
    f: Callable[[M, X], Tuple[float, M]],
    Batch: Axis,
    model: M,
    *inputs: X,
    per_device_parallelism: int,
    compute_axis_mapping: ResourceMapping,
    parameter_axis_mapping: ResourceMapping,
) -> Tuple[float, M]:
    """
    Accumulate gradients across a sharded dataset, keeping a local copy of the gradient on each row of the data
     parallel axis. (If the model is not sharded, then a copy of the gradient is on each individual device.)

     Parameters:
        f: a function that takes a model and a batch of inputs and returns a tuple of (loss, gradient)
        per_device_parallelism: how many examples to process at once on each device
        inputs: inputs with the batch axis. non-named arrays assume that the 0th axis is the batch axis.
        compute_axis_mapping: a ResourceMapping for doing compute. The model should be sharded this way
        parameter_axis_mapping: a ResourceMapping for doing parameter updates. The model should be sharded this way
    """
    batch_size = Batch.size
    with hax.axis_mapping(compute_axis_mapping):
        data_axis_size = hax.partitioning.physical_axis_size(Batch)
        if data_axis_size is None:
            raise ValueError(f"{Batch} axis must be sharded")
        physical_axis_name = hax.partitioning.physical_axis_name(Batch)
        assert physical_axis_name is not None

    # first things first, we want a copy of our gradient sharded like our model, along with a loss value
    loss = jnp.zeros(())
    grad = jax.tree_util.tree_map(jnp.zeros_like, model)
    grad = shard_with_axis_mapping(grad, parameter_axis_mapping)

    assert (
        batch_size % data_axis_size == 0
    ), f"batch size {batch_size} must be divisible by data axis size {data_axis_size}"
    microbatch_size = data_axis_size * per_device_parallelism
    assert (
        batch_size % microbatch_size == 0
    ), f"batch size {batch_size} must be divisible by microbatch size {microbatch_size}"

    num_micro_steps = batch_size // microbatch_size

    Microbatch = Axis(Batch.name, microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)

    assert num_micro_steps * microbatch_size == batch_size

    with hax.axis_mapping(compute_axis_mapping, merge=False):
        # second, we want to reshape our data to (num_micro_steps, micro_batch_size, ...), sharded along the data axis
        inputs = _reshape_for_microbatch(Batch, Microbatch, AccumStep, inputs)

        # third, we want to do compute.
        def loop(acc, microbatch):
            loss, grad = acc
            this_loss, this_grad = f(model, *microbatch)

            loss += this_loss
            grad = jax.tree_map(jnp.add, grad, this_grad)

            return loss, grad

        loss, grad = hax.fold(loop, AccumStep)((loss, grad), inputs)
        #grad = shard_with_axis_mapping(grad, parameter_axis_mapping)

    return loss / num_micro_steps, jax.tree_map(lambda x: x / num_micro_steps, grad)


@named_call
def _reshape_for_microbatch(Batch: Axis, Microbatch: Axis, AccumStep: Axis, inputs):
    def _reshape(x):
        if isinstance(x, hax.NamedArray):
            x = x.unflatten_axis(Batch, (AccumStep, Microbatch))
            return auto_sharded(x)
        elif isinstance(x, jnp.ndarray):
            x = x.reshape((AccumStep.size, Microbatch.size) + x.shape[1:])
            return with_sharding_constraint(x, PartitionSpec(None, ResourceAxis.DATA, *(None,) * (len(x.shape) - 2)))
        else:
            assert jnp.isscalar(x)
            return x

    return jax.tree_util.tree_map(_reshape, inputs, is_leaf=is_named_array)
