import dataclasses

import jax.numpy as jnp
import jmp
from jax.random import PRNGKey

from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def test_gradient_checkpointing():
    policy = jmp.get_policy("f32")

    # ensure that gradient checkpointing doesn't change the output
    # (this is a regression test for a bug that caused the output to change)
    for num_blocks in [1, 2, 4, 8, 12]:
        config = Gpt2Config(
            seq_len=16,
            hidden_dim=72,
            num_layers=num_blocks,
            num_heads=8,
            gradient_checkpointing=False,
        )
        config_checkpoint = dataclasses.replace(config, gradient_checkpointing=True)
        key = PRNGKey(0)

        vocab = Axis("vocab", 128)

        model = Gpt2LMHeadModel(vocab, config, key=key, mp=policy)
        model_checkpoint = Gpt2LMHeadModel(vocab, config_checkpoint, key=key, mp=policy)

        input_ids = jnp.arange(16, dtype=jnp.int32)

        a1 = model(input_ids, key)
        a2 = model_checkpoint(input_ids, key)

        assert jnp.all(jnp.isclose(a1, a2, rtol=1e-4, atol=1e-5)), f"failed with num_blocks={num_blocks}"
