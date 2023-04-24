import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear
import logging


from configs.q5_nature import config


class NatureQN(Linear):
    """
    Implementing DQN that will solve MinAtar's environments.

    Model configuration can be found in the assignment PDF, section 4a.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?



        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. To calculate the size of the input to the first linear layer, you
               can use online tools that calculate the output size of a
               convolutional layer (e.g. https://madebyollin.github.io/convnet-calculator/)
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()

        ##############################################################
        ################ YOUR CODE HERE - 20-30 lines ################
        # Q Network
        self.q_network = nn.Sequential(
            nn.Conv2d(n_channels*self.config.state_history, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

         # Target Network
        self.target_network = nn.Sequential(
            nn.Conv2d(n_channels*self.config.state_history, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Initialize target network weights to be the same as Q network weights
        self.target_network.load_state_dict(self.q_network.state_dict())

    def forward(self, x, network="q_network"):
        if network == "q_network":
            out = self.q_network(x)
        elif network == "target_network":
            out = self.target_network(x)
        else:
            raise ValueError("Network argument must be either 'q_network' or 'target_network'")

        return out

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = state.permute(0, 3, 1, 2)  # Swap channel and height/width dimensions
        state = state.float() / 255.0  # Normalize pixel values to range [0, 1]
        q_values = self.forward(state, network=network)
        return q_values
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
