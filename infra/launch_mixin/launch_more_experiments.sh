#!/bin/bash

config_dir="config/mixin"

config="$config_dir/512/gpt2_small_512_0.5.yaml"
echo "Running train_lm.py with $config"
gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
sleep 9000

seqlens=(64 128 256 512)
for seqlen in "${seqlens[@]}"; do
    config="$config_dir/${seqlen}/gpt2_small_${seqlen}_16.yaml"
    echo "Running train_lm.py with $config"
    gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
    sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds
done