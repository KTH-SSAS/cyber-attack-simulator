CONFIG=config/small_env_config.yaml
GPUS=1
STOP_ITER=300
NUM_WORKERS=0
ENV_PER_WORKER=15
BATCH_SIZE=12000

train:
	poetry run python ./scripts/run_rllib.py --config-file $(CONFIG) --wandb-sync --gpu-count $(GPUS) --stop-iter $(STOP_ITER) --num-workers $(NUM_WORKERS) --env-per-worker $(ENV_PER_WORKER) --batch-size $(BATCH_SIZE) --render

