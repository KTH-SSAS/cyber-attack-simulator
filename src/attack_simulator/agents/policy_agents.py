# import numpy as np
import torch

from .agent import Agent


# design influenced by https://github.com/pytorch/examples/tree/master/reinforcement_learning
class PolicyModel(torch.nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            # FIXME: temporarily disable Dropout, see
            #    https://gitr.sys.kth.se/pontusj/openai_attack_simulation/issues/27
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, num_actions),
            torch.nn.Softmax(dim=0),
        )

    def forward(self, state):
        return self.model(state)


class ReinforceAgent(Agent):
    def __init__(self, agent_config):
        if "gamma" not in agent_config:
            agent_config["gamma"] = 0.9
        if "use_cuda" not in agent_config:
            agent_config["use_cuda"] = False
        self.__dict__.update(agent_config)

        self.policy = PolicyModel(self.input_dim, self.num_actions, self.hidden_dim)

        if self.use_cuda:
            self.policy = self.policy.cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.training = True

        self._loss = -1
        self.rewards = []
        self.saved_log_probs = []

    def act(self, state):
        if self.use_cuda:
            state = torch.Tensor(state).cuda()  # pragma: no cover (untestable without CUDA HW)
        else:
            state = torch.Tensor(state)

        action_probabilities = self.policy.forward(state)
        # Create a distribution from the action probabilities...
        m = torch.distributions.Categorical(action_probabilities)
        action: torch.Tensor = m.sample()  # and sample an action from it.
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self, new_state, reward, done):
        self.rewards.append(reward)
        if done:
            loss = self._calculate_loss()
            if self.training:
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self._loss = loss.item()
            self.saved_log_probs = []
            self.rewards = []

    def _calculate_loss(self):  # , normalize_returns=False):
        # eps = np.finfo(np.float32).eps.item()  # for numerical stability
        R = 0  # Return at t=0
        num_steps = len(self.rewards)
        returns = torch.zeros(num_steps)  # Array to store returns
        loss = torch.zeros(num_steps)

        for i in reversed(range(num_steps)):
            R = self.rewards[i] + self.gamma * R
            returns[i] = R

        # if normalize_returns:
        #     if len(returns) > 1:
        #         returns = (returns - returns.mean()) / (returns.std() + eps)  # normalize returns
        #     else:
        #         returns = returns / returns  # blows up, if returns == 0

        for i, (log_prob, R) in enumerate(zip(self.saved_log_probs, returns)):
            loss[i] = -log_prob * R  # minus sign on loss for gradient ascent

        return loss.sum()

    def train(self, mode=True):
        self.training = mode
        self.policy.train(mode)

    @property
    def trainable(self):
        return True

    @property
    def loss(self):
        return self._loss
