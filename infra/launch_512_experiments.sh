#!/bin/bash

config_dir="config/mixin/512"
config_nums=(0.25 0.5 2 4 8)

for num in "${config_nums[@]}"; do
    config="$config_dir/gpt2_small_512_${num}.yaml"
    echo "Running train_lm.py with $config"
    gcloud compute tpus tpu-vm ssh kathli-32 --zone europe-west4-a --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 /home/kml11/levanter-midi/infra/run.sh python /home/kml11/levanter-midi/src/levanter/main/train_lm.py --config_path /home/kml11/levanter-midi/$config"
    sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds
done