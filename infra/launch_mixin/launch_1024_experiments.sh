# rerun 1024
config_1024 = "config/gpt2_small_1024_2hr.yaml"
echo "Running train_lm.py with $config_1024"
gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config_1024"
sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds

# rerun 512
config_512 = "config/mixin/512/gpt2_small_512_1.yaml"
echo "Running train_lm.py with $config_512"
gcloud compute tpus tpu-vm ssh kathli-nope-32 --zone us-east1-d --worker=all --command="WANDB_API_KEY=9dc3048e28645ac47904ed17c0f13bd782bdd3c7 levanter-midi/infra/launch.sh python levanter-midi/src/levanter/main/train_lm.py --config_path levanter-midi/$config_512"
sleep 9000 # 2.5 hours = 2.5 * 60 * 60 = 9000 seconds