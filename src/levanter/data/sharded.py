import functools
import itertools
from functools import cached_property
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, PyTree

import haliax as hax
import levanter.mesh
from haliax import NamedArray
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from levanter.data import Dataset
from levanter.data.dataset import ShardableDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape


In = TypeVar("In")
Ex = TypeVar("Ex")

# TODO: write tests to verify this works when data spans multiple processes

_TensorSliceIndex = Tuple[slice, ...]


class GlobalBatchDataset(Dataset[PyTree[jax.Array]]):
    """
    GlobalBatchDataset wraps a "local dataset" (a dataset that is shardable and can be iterated over) to produce
    distributed/sharded jax.Arrays representing batches of data. Each array that has a global shape
    but only has the data for some of the chunks of the array (namely, the ones on the local devices).
    Thus, each process loads the data for its devices.

    The details are a bit complex: We have a device mesh of shape (data, model). We want each row of the device mesh to
    get batch_size//num_rows examples. Usually, a process will be responsible for one or more entire rows, meaning
    that it wil load data that is distinct from every other process. However, if num_cols > num_devices_per_process,
    then some processes will need to load the same data. We use the process_mesh_position to determine which data to
    load, by determining which row(s) of the device mesh the process is responsible for.

    For now GlobalBatchDataset is restricted to datasets that return a single sequence of tokens.

    :arg local_dataset: a dataset that is shardable and can be iterated over
    :arg mesh: the device mesh
    :arg Batch: the batch size
    """

    def __init__(
        self,
        local_dataset: ShardableDataset[Sequence[int]],
        mesh: Mesh,
        Batch: hax.Axis,
        axis_resources: Optional[ResourceMapping] = None,
        *,
        override_process_data_pos: Optional[int] = None,  # for testing
        override_process_data_groups: Optional[int] = None,  # for testing
    ):
        self.mesh = mesh
        self.Batch = Batch
        self.axis_resources = axis_resources

        process_data_pos = override_process_data_pos or levanter.mesh.process_mesh_position(mesh)[0]
        num_data_process_groups = override_process_data_groups or levanter.mesh.process_mesh_size(mesh)[0]

        if not override_process_data_groups:
            assert num_data_process_groups <= jax.process_count()

        self.process_data_pos = process_data_pos
        self.num_data_process_groups = num_data_process_groups
        assert self.Batch.size % num_data_process_groups == 0

        self.local_dataset = local_dataset.shard(process_data_pos, num_data_process_groups)

    def __iter__(self) -> Iterator[PyTree[jax.Array]]:
        one_item_generator = iter(self.local_dataset)

        shape_leaves, shape_structure = jax.tree_util.tree_flatten(self.item_shape)

        for _ in range(self._global_min_length):
            # ok this is a bit messy: we want to create a batch of items from our dataset, only loading
            # the relevant data for each process.
            # In general an item is represented as a PyTree, whose leaves are (named or unnamed) arrays.
            # To make a batch we just want to add a leading dimension to each leaf array by stacking.
            # That is, we have (conceptually) a List[PyTree[Array]] and we want to produce a PyTree[List[Array]]
            # The difference is that we want to do this in a way that only loads the relevant data for each process
            # So it's more that we have a LocalBatch[PyTree[Array]] and we want to produce a PyTree[GlobalBatch[Array]]
            # because more than one device can get the same data, we need to make sure we only load it once since we're
            # streaming. This is the cache
            stacked_local_batches: Dict[
                Tuple[int, int], List[Union[Array, NamedArray]]
            ] = {}  # batch indices -> list of items

            batch_offset = self.process_data_pos * self.local_batch_size
            local_batch: List[PyTree] = list(itertools.islice(one_item_generator, self.local_batch_size))

            def get_local_batch(begin: int, end: int) -> List[Array]:
                assert begin >= batch_offset

                key = (begin, end)
                if key in stacked_local_batches:
                    return stacked_local_batches[key]

                individual_datums = local_batch[(begin - batch_offset) : (end - batch_offset)]

                with hax.enable_shape_checks(False):
                    device_batch = _stack_tree(self.Batch, individual_datums)
                batch_leaves = jax.tree_util.tree_leaves(device_batch)

                stacked_local_batches[key] = batch_leaves

                return batch_leaves

            # Callback passed to jax.make_array_from_callback to get the data for each device
            def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
                batch_slice = indices[0]
                begin, end, _ = batch_slice.indices(self.Batch.size)
                local_batch = get_local_batch(begin, end)
                leaf = local_batch[leaf_index]
                other_indices = indices[1:]
                if all(idx == slice(None) for idx in other_indices):
                    return leaf
                else:
                    return leaf[(..., *indices[1:])]

            def make_global_array_for_leaf(leaf_index, item_leaf_shape: Union[ShapeSpec, NamedShapeSpec]):
                raw_array = jax.make_array_from_callback(
                    to_raw_shape(item_leaf_shape),
                    jax.sharding.NamedSharding(self.mesh, self._pspec_for(item_leaf_shape)),
                    lambda indices: get_local_data_for_leaf(indices, leaf_index),
                )
                if isinstance(item_leaf_shape, NamedShapeSpec):
                    return hax.NamedArray(raw_array, item_leaf_shape.shape)
                else:
                    return raw_array

            # TODO: with a bit more fanciness, we can avoid needing the item_shape
            gda_leaves = [
                make_global_array_for_leaf(leaf_index, shape) for leaf_index, shape in enumerate(shape_leaves)
            ]

            gda_tree = jax.tree_util.tree_unflatten(shape_structure, gda_leaves)

            yield gda_tree  # type: ignore

    def _pspec_for(self, shape_spec: Union[ShapeSpec, NamedShapeSpec]) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.Batch, self.axis_resources)
            assert batch_name is not None
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.axis_resources)  # type: ignore

    @staticmethod
    def _get_begin_end_for_slice(tensor_shape, tslice_index) -> Tuple[Tuple[int, int], ...]:
        # begin, end, step
        my_indices: Tuple[Tuple[int, int, int], ...] = tuple(
            s.indices(axis_size) for axis_size, s in zip(tensor_shape, tslice_index)
        )
        assert all(s[2] == 1 for s in my_indices)  # ensure step is 1
        return tuple(s[0:2] for s in my_indices)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        def _batchify_shape_spec(shape_spec: Union[ShapeSpec, NamedShapeSpec]):
            shape = shape_spec.shape
            assert shape is not None, "item_shape must have a fully determined shape to work with batching"
            if isinstance(shape_spec, NamedShapeSpec):
                return NamedShapeSpec((self.Batch,) + shape, shape_spec.dtype)
            else:
                return ShapeSpec((self.Batch.size,) + shape, shape_spec.dtype)

        return jax.tree_map(_batchify_shape_spec, self.local_dataset.item_shape)

    def __len__(self):
        return self._global_min_length

    @property
    def batch_size(self) -> int:
        """Returns the 'global' batch size: the effective number of examples in a batch across all devices/hosts"""
        return self.Batch.size

    @property
    def local_batch_size(self) -> int:
        """Returns the 'local' batch size: the number of examples in a batch on this host"""
        return self.batch_size // self.num_data_process_groups

    @cached_property
    def _global_min_length(self):
        # TODO: to test this effectively we'll need to set up a test harness across a multinode instance
        # length is the min over the shards, so we have to communicate the min via jax
        local_len = len(self.local_dataset) // self.local_batch_size
        all_lengths = process_allgather(jnp.array(local_len))
        return int(jnp.min(all_lengths))

    def _stack_leaves_unchecked(self, *leaves):
        assert len(leaves) <= self.Batch.size
        assert self.Batch.size % len(leaves) == 0

        if is_named_array(leaves[0]):
            with hax.enable_shape_checks(False):  # because we're building parts of the array on each device
                return hax.stack(self.Batch, leaves)
        else:
            return np.stack(leaves)


@functools.partial(jax.jit, static_argnums=(0,))
def _stack_tree(batch, individual_datums):
    def _stack_leaves_unchecked(*leaves):
        if is_named_array(leaves[0]):
                return hax.stack(batch, leaves)
        else:
            return jnp.stack(leaves)

    return jax.tree_map(_stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)
