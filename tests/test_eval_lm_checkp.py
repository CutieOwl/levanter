import os
import tempfile

import jax
import ray

import haliax

import levanter.main.eval_lm as eval_lm
import gs_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.distributed import RayConfig
from levanter.logging import WandbConfig
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.utils.py_utils import logical_cpu_core_count


def setup_module(module):
    ray_designated_cores = max(1, logical_cpu_core_count())
    ray.init("local", num_cpus=ray_designated_cores)


def teardown_module(module):
    ray.shutdown()


def test_eval_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = eval_lm.Gpt2Config(
        num_layers=12,
        num_heads=12,
        seq_len=[1024],
        hidden_dim=768,
    )

    with tempfile.TemporaryDirectory() as f:
        try:
            data_config = gs_test_corpus.gs_corpus_config(f)
            tok = data_config.the_tokenizer
            Vocab = haliax.Axis("vocab", len(tok))
            model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

            config = eval_lm.EvalLmConfig(
                data=data_config,
                model=model_config,
                trainer=eval_lm.TrainerConfig(
                    per_device_eval_parallelism=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
                checkpoint_path=f"/jagupard31/scr1/kathli/checkpoints/765gavzf/step-10200/",
            )
            eval_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


if __name__ == "__main__":
    test_eval_lm()