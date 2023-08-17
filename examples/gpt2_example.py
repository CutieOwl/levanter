import logging
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import pyrallis
from jax.interpreters.pxla import PartitionSpec
from transformers import GPT2Tokenizer

import haliax as hax
import haliax.random
import wandb
from haliax import Axis
from haliax.partitioning import ResourceAxis, named_pjit, round_axis_for_partitioning
from haliax.jax_utils import maybe_rng_split
from levanter import callbacks
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset
from levanter.data.text import CachedLMDatasetConfig, TokenSeqDataset
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.jax_utils import global_key_array, parameter_count
from levanter.logging import capture_time, log_time_to_wandb
from levanter.modeling_utils import cross_entropy_loss_and_log_normalizers
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo, TrainerHooks
from py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data_short: CachedLMDatasetConfig = CachedLMDatasetConfig()
    data_long: CachedLMDatasetConfig = CachedLMDatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    log_z_regularization: float = 0.0
    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.trainer.initialize(config)
    print("qq")

    tokenizer: GPT2Tokenizer = config.data_short.the_tokenizer

    # some axes we need
    short_Batch = Axis("batch", config.trainer.short_train_batch_size)
    EvalBatch = Axis("batch", config.trainer.eval_batch_size)
    long_Batch = Axis("batch", config.trainer.long_train_batch_size)

    short_SeqLen = Axis(name="seqlen", size=config.trainer.short_seq_len)
    short_KeySeqLen = Axis(name="key_seqlen", size=config.trainer.short_seq_len)
    long_SeqLen = Axis(name="seqlen", size=config.trainer.long_seq_len)
    long_KeySeqLen = Axis(name="key_seqlen", size=config.trainer.long_seq_len)

    # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
    # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    short_dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data_short.build_or_load_document_cache("train"), config.model.seq_len),
        config.trainer.short_device_mesh,
        short_Batch,
        compute_axis_mapping,
    )

    short_eval_dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data_short.build_or_load_document_cache("validation"), config.model.seq_len),
        config.trainer.short_device_mesh,
        EvalBatch,
        compute_axis_mapping,
    )

    long_dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data_long.build_or_load_document_cache("train"), config.model.seq_len),
        config.trainer.long_device_mesh,
        long_Batch,
        compute_axis_mapping,
    )

    with config.trainer.short_device_mesh as short_mesh, config.trainer.long_device_mesh as long_mesh:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Mixed Precision: We use the "jmp" library to handle mixed precision training. It basically has three dtypes:
        # 1) compute (typically bfloat16)
        # 2) parameter (typically float32)
        # 3) output (sometimes float32)
        # I like to think of these as "semantic" dtypes: compute is the dtype we do most of our math in, parameter is
        # the dtype we store our parameters in, and output is the dtype we use for loss calculations.
        mp: jmp.Policy = config.trainer.mp

        # initialize the model
        # This function
        # 1) initializes model weights
        # 2) ensures all model weights are the right dtype
        # 3) ensures the model is partitioned across the mesh according to the parameter_axis_mapping
        @named_pjit(axis_resources=parameter_axis_mapping)
        def init_model():
            model = Gpt2LMHeadModel(Vocab, config.model, key=model_key)
            return mp.cast_to_param(model)

        model = init_model()

        wandb.summary["parameter_count"] = parameter_count(model)

        # initialize the optimizer
        # This is basically the same as the model.
        optimizer = config.trainer.optimizer()
        opt_state = named_pjit(optimizer.init, axis_resources=parameter_axis_mapping)(model)

        # masks for attention and loss
        def attention_mask(inference, fcm_key, SeqLen_axis, KeySeqLen_axis):
            causal_mask = hax.nn.attention.causal_mask(SeqLen_axis, KeySeqLen_axis)

            # forgetful causal masking
            if not inference and config.fcm_prob > 0:
                fcm_mask = hax.nn.attention.forgetful_causal_mask(KeySeqLen_axis, config.fcm_prob, key=fcm_key)
                causal_mask = causal_mask & fcm_mask
            return causal_mask

        # don't want to compute the loss w.r.t. the final token
        short_loss_mask = 1 - hax.nn.one_hot(-1, short_SeqLen, dtype=jnp.float32)  # one everywhere except the last token
        long_loss_mask = 1 - hax.nn.one_hot(-1, long_SeqLen, dtype=jnp.float32)  # one everywhere except the last token

        # loss function: this computes the loss with respect to a single example
        def compute_loss(model: Gpt2LMHeadModel, input_ids, attn_mask, key, inference, SeqLen_axis, loss_mask):
            print("compute_loss input_ids", input_ids)
            print("compute_loss key", key)
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask, key=key, inference=inference, SeqLen_axis=SeqLen_axis)
                pred_y = mp.cast_to_output(pred_y)

                # need to roll the target tokens back by one so that each token is predicting the next token
                target_y = haliax.roll(input_ids, -1, SeqLen_axis)
                target_y = haliax.nn.one_hot(target_y, Vocab, dtype=pred_y.dtype)

                loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)
                loss = hax.mean(loss, where=loss_mask)

                if not inference and config.log_z_regularization > 0:
                    logz_mse = hax.mean((log_normalizers**2))
                    loss += config.log_z_regularization * logz_mse

                return loss.scalar()

        def short_train_batch_loss(model, input_ids, attn_mask, key):
            print("tbl input_ids", input_ids)
            print("tbl key", key)
            return compute_loss(model, input_ids, attn_mask, key, inference=False, SeqLen_axis=short_SeqLen, loss_mask=short_loss_mask)
        
        def long_train_batch_loss(model, input_ids, attn_mask, key):
            print("tbl input_ids", input_ids)
            print("tbl key", key)
            return compute_loss(model, input_ids, attn_mask, key, inference=False, SeqLen_axis=long_SeqLen, loss_mask=long_loss_mask)

        # training loop
        # donate args to conserve memory
        @named_pjit(axis_resources=parameter_axis_mapping, donate_args=True)
        def short_train_step(model, opt_state, input_ids, key):
            if key is not None:
                mask_key, key = jrandom.split(key)
                mask_keys = maybe_rng_split(mask_key, short_Batch.size)
            else:
                mask_keys = None

            attn_mask = hax.vmap(attention_mask, short_Batch)(False, mask_keys, short_SeqLen, short_KeySeqLen)
            attn_mask = hax.auto_sharded(attn_mask)

            #print("train_step input_ids", input_ids)
            #print("keys", key)
            loss, grads = accumulate_gradients_sharded(
                eqx.filter_value_and_grad(short_train_batch_loss),
                short_Batch,
                model,
                input_ids,
                attn_mask,
                key=key,
                per_device_parallelism=config.trainer.short_per_device_parallelism,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            # distribute gradients across the mesh and apply them
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state
        
        @named_pjit(axis_resources=parameter_axis_mapping, donate_args=True)
        def long_train_step(model, opt_state, input_ids, key):
            if key is not None:
                mask_key, key = jrandom.split(key)
                mask_keys = maybe_rng_split(mask_key, long_Batch.size)
            else:
                mask_keys = None

            attn_mask = hax.vmap(attention_mask, long_Batch)(False, mask_keys, long_SeqLen, long_KeySeqLen)
            attn_mask = hax.auto_sharded(attn_mask)

            print("train_step input_ids", input_ids)
            print("keys", key)
            loss, grads = accumulate_gradients_sharded(
                eqx.filter_value_and_grad(long_train_batch_loss),
                long_Batch,
                model,
                input_ids,
                attn_mask,
                key=key,
                per_device_parallelism=config.trainer.long_per_device_parallelism,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            # distribute gradients across the mesh and apply them
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # evaluation loss and loop

        @named_pjit(axis_resources=parameter_axis_mapping)
        def eval_loss(model, input_ids):
            # we just use the short model to evaluate, don't eval on the long model
            input_ids = hax.named(input_ids, (EvalBatch, short_SeqLen))
            # just use causal mask for evaluation
            mask = hax.nn.attention.causal_mask(short_SeqLen, short_KeySeqLen)
            return compute_loss(model, input_ids, mask, None, True, short_SeqLen, short_loss_mask)

        # Set up evaluation
        def evaluate_step(info: StepInfo):
            # we just use the short model to evaluate, don't eval on the long model
            with hax.axis_mapping(compute_axis_mapping):
                # standard evaluation loop
                loss = 0.0
                n = 0

                for batch in short_eval_dataset:
                    loss += eval_loss(model, batch).item()
                    n += 1

                if n > 0:
                    loss /= n

            logger.info(f"validation loss: {loss:.3f}")
            if wandb.run is not None:
                wandb.log({"eval/loss": loss}, step=info.step)

            return loss

        # boilerplate hooks and such
        engine = TrainerHooks()
        engine.add_hook(callbacks.pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(callbacks.log_to_wandb, every=1)
        engine.add_hook(
            callbacks.log_performance_stats(config.trainer.short_seq_len, config.trainer.short_train_batch_size), every=1
        )
        engine.add_hook(evaluate_step, every=config.trainer.steps_per_eval)
        engine.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_name)
        engine.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency
        engine.add_hook(lambda x: callbacks.defragment(), every=100)

        # data loader
        short_iter_data = non_caching_cycle(short_dataset)
        long_iter_data = non_caching_cycle(long_dataset)

        # load the last checkpoint and resume if we want
        resume_step = None
        if config.trainer.load_last_checkpoint:
            checkpoint = checkpointer.load_checkpoint(
                model,
                (opt_state, training_key),
                config.trainer.load_checkpoint_path,
            )

            if checkpoint is not None:
                model, (opt_state, training_key), resume_step = checkpoint
                assert training_key.shape == jrandom.PRNGKey(0).shape
            elif config.trainer.load_checkpoint_path:
                raise ValueError("No checkpoint found")
            else:
                logger.info("No checkpoint found. Starting from scratch")

        # NOTE: removing the following for finetuning so we resume at step 0
        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: iter_data.seek(resume_step +1)
            import tqdm

            for i in tqdm.tqdm(range(resume_step + 1), desc="seeking data for resume"):
                if i % 8 == 7:
                    next(long_iter_data)
                else:
                    next(short_iter_data)
            resume_step = resume_step + 1
        else:
            resume_step = 0
        
            
        # finally, run the training loop
        for step in range(resume_step, config.trainer.num_train_steps):
            with capture_time() as step_time:
                if step % 8 == 7:
                    with log_time_to_wandb("throughput/loading_time", step=step):
                        input_ids = next(long_iter_data)
                        input_ids = hax.named(input_ids, (long_Batch, long_SeqLen))
                        #print("training_key", training_key)
                        #print("input_ids step", input_ids)
                        my_key, training_key = jrandom.split(training_key, 2)
                    step_loss, model, opt_state = long_train_step(model, opt_state, input_ids, my_key)
                    step_loss = step_loss.item()
                else:
                    with log_time_to_wandb("throughput/loading_time", step=step):
                        input_ids = next(short_iter_data)
                        input_ids = hax.named(input_ids, (short_Batch, short_SeqLen))
                        #print("training_key", training_key)
                        #print("input_ids step", input_ids)
                        my_key, training_key = jrandom.split(training_key, 2)
                    step_loss, model, opt_state = short_train_step(model, opt_state, input_ids, my_key)
                    step_loss = step_loss.item()

            with log_time_to_wandb("throughput/hook_time", step=step):
                engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        last_step = StepInfo(
            config.trainer.num_train_steps,
            model,
            opt_state,
            step_loss,
            training_key,
            step_duration=step_time(),
        )

        evaluate_step(last_step)
        checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    main()
