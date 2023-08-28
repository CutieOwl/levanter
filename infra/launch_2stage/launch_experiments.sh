#!/bin/bash

config_dir="config/2stage"
ratios=(1 2 4 8 0.5 0.25)
seq_lens=(512 256 128 64)

for ratio in "${ratios[@]}"; do
    for seq_len in "${seq_lens[@]}" do
        config="$config_dir/${seq_len}/gpt2_small_${seq_len}_${ratio}.yaml"
        echo "Running train_lm.py with $config"
        gcloud compute tpus tpu-vm ssh kathli-32 --zone europe-west4-a --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
        sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds
    done
done
