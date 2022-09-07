CONFIG=config/maze_env_config.yaml
GPUS=1
STOP_ITER=300
NUM_WORKERS=0
ENV_PER_WORKER=15
BATCH_SIZE=12000

BASE_CMD = poetry run python ./scripts/run_rllib.py

train-cuda: 
	$(BASE_CMD) --config-file $(CONFIG) --wandb-sync --gpu-count $(GPUS) --stop-iter $(STOP_ITER) --num-workers $(NUM_WORKERS)

train:
	$(BASE_CMD) --config-file $(CONFIG) --wandb-sync --stop-iter $(STOP_ITER) --num-workers $(NUM_WORKERS)

local:
	$(BASE_CMD) --config-file $(CONFIG) --stop-iter $(STOP_ITER) --local