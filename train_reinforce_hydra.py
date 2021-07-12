from attack_simulator.utils import Runner
import logging
import numpy.random as random
import torch
import hydra 
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig):
    
    #print(OmegaConf.to_yaml(cfg))

    #logging.getLogger("simulator").setLevel(logging.DEBUG)
    #logging.getLogger("simulator").addHandler(logging.FileHandler("simulator.log", mode="w"))
    #logging.getLogger("trainer").setLevel(logging.DEBUG)
    #logging.getLogger("trainer").addHandler(logging.FileHandler("trainer.log", mode="w"))

    if cfg.env.graph_size == 'small':
        attack_steps = 7
    elif cfg.env.graph_size == 'medium':
        attack_steps = 29
    else:
        attack_steps = 78

    if cfg.deterministic:
        random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)

    services = 18
    include_services_in_state = cfg.trainer.include_services
    if include_services_in_state:
        input_dim = attack_steps + services
    else:
        input_dim = attack_steps

    runner = Runner(cfg.agent.type, cfg.deterministic,  cfg.env.early_flag_reward, cfg.env.late_flag_reward,
                                             cfg.env.final_flag_reward, cfg.env.easy_ttc, cfg.env.hard_ttc, cfg.env.graph_size,
                                             cfg.env.attacker_strategy, cfg.env.true_positive, cfg.env.false_positive,
                                             input_dim, services, cfg.agent.hidden_width, cfg.agent.lr, cfg.trainer.allow_skips, cfg.trainer.include_services)

    runner.train_and_evaluate(cfg.trainer.n_simulations, cfg.trainer.evaluation_rounds)

if __name__ == '__main__':
	main()