import tempfile

import equinox as eqx
import jax
import jmp
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp
from jax.random import PRNGKey

import sys
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/transformers-levanter/src')
sys.path.append('/afs/cs.stanford.edu/u/kathli/repos/levanter-midi/src')

from transformers.models.gpt2 import GPT2LMHeadModel as HfGpt2LMHeadModel
from transformers import GPT2Config as HfGpt2Config
#from utils import skip_if_no_torch

import haliax as hax
from haliax import Axis, NamedArray
from haliax.util import is_named_array
from haliax.partitioning import ResourceAxis, named_pjit, round_axis_for_partitioning

from levanter.config import TrainerConfig
from levanter.modeling_utils import cross_entropy_loss
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore


#@skip_if_no_torch
def test_hf_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("gpt2", None)


#@skip_if_no_torch
def test_mistral_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("/nlp/scr/kathli/checkpoints/whole-water-182/step-6627/hf", "/nlp/scr/kathli/checkpoints/whole-water-182/step-6627", None)


def _rand_input(key: PRNGKey, seq_len: int, vocab_size) -> jnp.ndarray:
    return jrandom.randint(key, (seq_len,), 0, vocab_size)

def get_gpt2_model(checkpoint_path):
    vocab_size = 55028 #len(tokenizer)
    Vocab = Axis("vocab", vocab_size)
    key = jax.random.PRNGKey(0)

    config_model = Gpt2Config()
    with jax.default_device(jax.devices("cpu")[0]):
        model = Gpt2LMHeadModel(Vocab, config_model, key=key)

        with hax.enable_shape_checks(False):
            model = tree_deserialize_leaves_tensorstore(f"{checkpoint_path}/model", model)

        def patch_vocab(array):
            if is_named_array(array):
                return patch_vocab_size(array.array, array)
            else:
                return array

        #model = jax.tree_util.tree_map(patch_vocab, model, is_leaf=is_named_array)
        return model


def patch_vocab_size(inner: jnp.ndarray, like: NamedArray):
    # for partitioning reasons we frequently round the vocab size, but we need to patch it back
    # to the original size for the HF checkpoint to work
    if any(ax.name == "vocab" for ax in like.axes):
        index_of_vocab = next(i for i, ax in enumerate(like.axes) if ax.name == "vocab")
        desired_vocab_size = like.axes[index_of_vocab].size
        vocab_size = inner.shape[index_of_vocab]
        if vocab_size != desired_vocab_size:
            logger.info(f"Patching vocab size from {vocab_size} back to {desired_vocab_size} for HF checkpoint")
            inner = jnp.take(inner, jnp.arange(desired_vocab_size), axis=index_of_vocab)
    return inner


def print_closeness(model, converted_model, prefix):
    # TODO: assert compatibility of old and new values (type, shape, etc.)
    # print("tree", tree)
    if isinstance(converted_model, NamedArray):
        print("Compare", prefix, ":", jnp.allclose(model.array, converted_model.array))
    elif isinstance(model, eqx.Module):
        for attr in dir(model):
            if not attr.startswith('__'):
                print_closeness(getattr(model, attr), getattr(converted_model, attr), f'{prefix}.{attr}')


def _roundtrip_compare_gpt2_checkpoint(hf_model_path, lev_model_path, revision):
    import torch

    from levanter.compat.hf_checkpoints import (
        load_hf_gpt2_checkpoint,
        load_hf_model_checkpoint,
        save_hf_gpt2_checkpoint,
    )

    config, data = load_hf_model_checkpoint(hf_model_path) #, revision=revision
    config = HfGpt2Config.from_dict(config)
    print("n_positions", config.n_positions)
    # torch model is the model created from converting original model checkpoint to torch
    torch_model: HfGpt2LMHeadModel = HfGpt2LMHeadModel.from_pretrained(hf_model_path, config=config) #, revision=revision
    torch_model.eval()

    # model is the original levanter model checkpoint
    model = get_gpt2_model(lev_model_path)

    input = hax.random.randint(PRNGKey(0), model.SeqLen, 0, model.Vocab.size)

    # convert torch model back to levanter model
    converted_model = load_hf_gpt2_checkpoint(hf_model_path)

    # we should have model == converted_model.
    #print("model", model)
    #print("converted model", converted_model)
    #print("token_embeddings:", model.embeddings.token_embeddings)
    #print("converted token_embeddings:", converted_model.embeddings.token_embeddings.array)
    print_closeness(model, converted_model, '')

    # we compare softmaxes because the numerics are wonky and we usually just care about the softmax
    torch_out = torch_model(torch.from_numpy(onp.array(input.array)).to(torch.int32).unsqueeze(0))
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    print("torch logits", torch_out)
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    KeySeqLen = model.SeqLen.alias(f"key_{model.SeqLen.name}")
    attn_mask = hax.nn.attention.causal_mask(model.SeqLen, KeySeqLen)

    Batch = Axis("batch", 1)
    input_with_batch = NamedArray(input.array.reshape((1, model.SeqLen.size)), (Batch, model.SeqLen))

    def compute(input):
        logits = model(input, inference=True, key=None, attn_mask=None)
        print("lev logits", logits)
        return hax.nn.softmax(logits, axis=model.Vocab)

    #compute = jax.jit(compute)
    print("input axes", input_with_batch.axes)
    jax_out = compute(input_with_batch).array
    jax_out = jax_out[0,:,:]
    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert onp.isclose(torch_out, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"


# Gradient tests


#@skip_if_no_torch
def test_hf_gradient():
    _compare_gpt2_checkpoint_gradients("gpt2", None)


def _compare_gpt2_checkpoint_gradients(model_id, revision):
    import torch

    from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint, load_hf_model_checkpoint

    config, data = load_hf_model_checkpoint(model_id, revision=revision)
    config = HfGpt2Config.from_dict(config)
    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_id, config=config, revision=revision)
    torch_model.eval()

    model = load_hf_gpt2_checkpoint(model_id, revision=revision)

    input = hax.random.randint(PRNGKey(0), model.SeqLen, 0, model.Vocab.size)

    def torch_loss(model, input_ids) -> torch.Tensor:
        return model(input_ids, labels=input_ids)[0]

    torch_out = torch_loss(torch_model, torch.from_numpy(onp.array(input.array)).to(torch.int64).unsqueeze(0))

    # don't want to compute the mask w.r.t. the final token
    loss_mask = hax.nn.one_hot(-1, model.SeqLen, dtype=jnp.float32)
    loss_mask = 1 - loss_mask  # one everywhere except the last token

    causal_mask = hax.nn.attention.causal_mask(model.config.SeqLen, model.config.KeySeqLen)

    def compute_loss(model, input_ids):
        pred_y = model(input_ids, key=None, inference=True, attn_mask=causal_mask)

        # need to roll the target tokens back by one so that each token is predicting the next token
        target_y = hax.roll(input_ids, -1, model.SeqLen)
        target_y = hax.nn.one_hot(target_y, model.Vocab, dtype=pred_y.dtype)

        token_loss = hax.mean(cross_entropy_loss(pred_y, model.Vocab, target_y), where=loss_mask)

        return token_loss.scalar()

    jax_compute_grad = jax.value_and_grad(compute_loss)
    jax_loss, jax_grad = jax_compute_grad(model, input)

    # gradients are kind of a pain to get at in torch, but we do it anyway
    torch_out.backward()
    torch_dict = torch_model.transformer.state_dict(keep_vars=True)
    torch_dict = {k: v.grad for k, v in torch_dict.items()}

    jax_grad: Gpt2LMHeadModel

    jax_grad_dict = jax_grad.to_torch_dict()

    for jax_key, jax_g in jax_grad_dict.items():
        if jax_key not in torch_dict:
            assert jax_key == "token_out_embeddings"
            continue

        torch_g = torch_dict[jax_key]
        assert onp.isclose(jax_g, torch_g.detach().cpu().numpy(), rtol=1e-2, atol=1e-2).all(), f"{jax_g} != {torch_g}"

    # now we also want to check that the optimizers do similar things
    trainer_config = TrainerConfig(weight_decay=0.0, learning_rate=1e-3, warmup_ratio=0.0)

    if trainer_config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), trainer_config.max_grad_norm)
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        betas=(trainer_config.beta1, trainer_config.beta2),
        eps=trainer_config.epsilon,
    )

    torch_optimizer.step()

    jax_optimizer = trainer_config.optimizer()
    state = jax_optimizer.init(model)
    updates, state = jax_optimizer.update(updates=jax_grad, state=state, params=model)
    new_model = eqx.apply_updates(model, updates)

    new_model_dict = new_model.to_torch_dict()
    torch_dict = torch_model.transformer.state_dict(keep_vars=True)

    # now compare new params
    for key, jax_p in new_model_dict.items():
        if key not in torch_dict:
            assert key == "token_out_embeddings"
            continue
        torch_p = torch_dict[key]
        assert onp.isclose(
            jax_p, torch_p.detach().cpu().numpy(), rtol=1e-3, atol=2e-3
        ).all(), f"{key}: {onp.linalg.norm(jax_p - torch_p.detach().cpu().numpy(), ord=onp.inf)}"

if __name__ == "__main__":
    test_mistral_gpt2_roundtrip()