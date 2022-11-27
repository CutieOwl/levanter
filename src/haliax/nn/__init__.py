import functools
from typing import Optional, Union

import jax.nn as jnn
import jax.numpy as jnp

import haliax
import haliax.nn.attention as attention

from ..core import NamedArray
from ..types import Axis, AxisSpec
from ..wrap import unwrap_namedarrays, wrap_axiswise_call, wrap_elemwise_unary, wrap_reduction_call
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm


relu = wrap_elemwise_unary(jnn.relu)
relu6 = wrap_elemwise_unary(jnn.relu6)
sigmoid = wrap_elemwise_unary(jnn.sigmoid)
softplus = wrap_elemwise_unary(jnn.softplus)
soft_sign = wrap_elemwise_unary(jnn.soft_sign)
silu = wrap_elemwise_unary(jnn.silu)
swish = wrap_elemwise_unary(jnn.swish)
log_sigmoid = wrap_elemwise_unary(jnn.log_sigmoid)
leaky_relu = wrap_elemwise_unary(jnn.leaky_relu)
hard_sigmoid = wrap_elemwise_unary(jnn.hard_sigmoid)
hard_silu = wrap_elemwise_unary(jnn.hard_silu)
hard_swish = wrap_elemwise_unary(jnn.hard_swish)
hard_tanh = wrap_elemwise_unary(jnn.hard_tanh)
elu = wrap_elemwise_unary(jnn.elu)
celu = wrap_elemwise_unary(jnn.celu)
selu = wrap_elemwise_unary(jnn.selu)
gelu = wrap_elemwise_unary(jnn.gelu)


def glu(x: NamedArray, axis: Axis) -> NamedArray:
    axis_index = x.axes.index(axis)
    return jnn.glu(x.array, axis_index)


logsumexp = wrap_reduction_call(jnn.logsumexp, False, supports_where=False)

# TODO: support where in softmax, etc
softmax = wrap_axiswise_call(jnn.softmax, False)
log_softmax = wrap_axiswise_call(jnn.log_softmax, False)


@functools.wraps(jnn.standardize)
def standardize(
    x: NamedArray,
    axis: AxisSpec,
    *,
    mean: Optional[NamedArray] = None,
    variance: Optional[NamedArray] = None,
    epsilon: float = 1e-5,
    where: Optional[NamedArray] = None,
) -> NamedArray:
    x, mean, variance, where = haliax.broadcast_arrays(x, mean, variance, where)  # type: ignore
    raw_x, mean, variance, where = unwrap_namedarrays(x, mean, variance, where)
    axis_indices = x._lookup_indices(axis)

    plain = jnn.standardize(raw_x, axis_indices, mean=mean, variance=variance, epsilon=epsilon, where=where)
    return NamedArray(plain, x.axes)


@functools.wraps(jnn.one_hot)
def one_hot(x: Union[NamedArray, int], class_axis: Axis, *, dtype=jnp.float_) -> NamedArray:
    if isinstance(x, NamedArray):
        array = jnn.one_hot(x.array, num_classes=class_axis.size, dtype=dtype)
        return NamedArray(array, x.axes + (class_axis,))
    else:
        assert isinstance(x, int)
        assert class_axis.size > x >= -class_axis.size

        array = jnp.zeros(class_axis.size, dtype=dtype).at[x].set(1)
        return haliax.named(array, class_axis)


__all__ = [
    "attention",
    "relu",
    "relu6",
    "sigmoid",
    "softplus",
    "soft_sign",
    "silu",
    "swish",
    "log_sigmoid",
    "leaky_relu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "elu",
    "celu",
    "selu",
    "gelu",
    "logsumexp",
    "softmax",
    "log_softmax",
    "one_hot",
    "Dropout",
    "LayerNorm",
    "Linear",
]
