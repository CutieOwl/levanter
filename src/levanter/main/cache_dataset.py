import logging
import os
from dataclasses import dataclass, field

import levanter
from levanter.data.shard_cache import LoggingMetricsMonitor, RichMetricsMonitor, build_cache
from levanter.data.text import BatchTokenizer, LMDatasetConfig
from levanter.distributed import RayConfig
from levanter.logging import init_logging
from levanter.tracker import NoopConfig, TrackerConfig


logger = logging.getLogger(__name__)


@dataclass
class RayCachedLMDatasetConfig(LMDatasetConfig, RayConfig):
    tracker: TrackerConfig = field(default_factory=NoopConfig)


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    init_logging(".", "cache_dataset.log")
    args.initialize()

    tokenizer = args.the_tokenizer

    for split in ["train", "validation"]:
        print(f"Caching {split} to {args.cache_dir}.")
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=args.enforce_eos)
        split_cache_dir = os.path.join(args.cache_dir, split)
        source = args.get_shard_source(split)

        if source is None:
            logger.warning(f"Skipping {split} because it is empty.")
            continue

        monitors = [RichMetricsMonitor(source.num_shards), LoggingMetricsMonitor("preprocess/" + split, commit=True)]

        cache = build_cache(
            cache_dir=split_cache_dir,
            input_shards=source,
            processor=batch_tokenizer,
            rows_per_chunk=args.rows_per_chunk,
            await_finished=False,
            monitors=monitors,
        )

        cache.await_finished()
        print(f"Finished caching {split} to {split_cache_dir}.")


if __name__ == "__main__":
    main()
