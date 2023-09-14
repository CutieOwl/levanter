#!/bin/bash

config="$config_dir/gpt2_small_1024_2hr.yaml"
echo "Running train_lm.py with $config"
gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
sleep 9000

config="$config_dir/128/gpt2_small_128_0.5.yaml"
echo "Running train_lm.py with $config"
gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
sleep 9000

config_dir="config/2stage"

ratios=(2 4 8 0.5)
for ratio in "${ratios[@]}"; do
    config="$config_dir/512/gpt2_small_512_${ratio}.yaml"
    echo "Running train_lm.py with $config"
    gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config"
    sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds
done