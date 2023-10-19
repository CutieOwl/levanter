import json
import os

from levanter.data.text import LMDatasetConfig

def gs_corpus_config(path):
    return LMDatasetConfig(
        train_urls=[f"gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.1-of-128.jsonl"],
        validation_urls=[f"gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.1-of-8.jsonl.gz"],
        cache_dir=f"gs://levanter-data/tokenized/openwebtext/",
    )
