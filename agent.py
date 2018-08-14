import argparse
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import config
from environment import Environment
from replay_memory import ReplayMemory,Transition
from numpy import random
import pickle
from dataset import Dataset
from model import AttnDecoder,AttnEncoder,DQN
from trainer import getArgParser
import numpy as np
steps_done = 0
class Agent:
    def __init__(self,time_step, split, lr):
        self.dataset = Dataset(T = time_step, split_ratio=split,binary_file=config.BINARY_DATASET)
        self.policy_net_encoder = AttnEncoder(input_size=self.dataset.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=time_step)
        self.policy_net_decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=time_step)
        self.policy_net = DQN(self.policy_net_encoder,self.policy_net_decoder)
        self.target_net_encoder = AttnEncoder(input_size=self.dataset.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=time_step)
        self.target_net_decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=time_step)
        self.target_net = DQN(self.target_net_encoder,self.target_net_decoder)
        if torch.cuda.is_available():
            self.policy_net_encoder = self.policy_net_encoder.cuda()
            self.policy_net_decoder = self.policy_net_decoder.cuda()
            self.target_net_encoder = self.target_net_encoder.cuda()
            self.target_net_decoder = self.target_net_decoder.cuda()
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        self.memory = ReplayMemory(config.MEMORY_CAPACITY);
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=lr)

    def select_action(self,state,test = False):
        global steps_done
        sample = random.random()
        eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1.*steps_done/config.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold or test == True:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            if torch.cuda.is_available():
                return torch.tensor([[random.randint(3)]], dtype=torch.long).cuda()
            else:
                return torch.tensor([[random.randint(3)]], dtype=torch.long)
    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = tuple([torch.cat(tuple([batch.state[i][j] for i in range(config.BATCH_SIZE)])) for j in range(3)])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = tuple([torch.cat(tuple([batch.next_state[i][j] for i in range(config.BATCH_SIZE)])) for j in range(3)])
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)
        next_state_values= self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def load_model(self,encoder_path = None,decoder_path=None,DQN_path = None):
        if(DQN_path != None):
            self.policy_net.load_state_dict(torch.load(DQN_path, map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.policy_net_encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.policy_net_decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
            self.policy_net = DQN(self.policy_net_encoder,self.policy_net_decoder)
            self.target_net.load_state_dict(self.policy_net.state_dict())
    def train(self,num_epochs,interval):
        env = Environment(np.array([0.5,0.5]))
        episode = 0
        for epoch in range(num_epochs):
            env.reset()
            state = (env.x[env.current_step].unsqueeze(0),env.y_seq[env.current_step].unsqueeze(0),env.position.unsqueeze(0))
            while(1):
                action = self.select_action(state)
                _,next_state,reward = env.step(action.item())
                if(next_state == None):
                    break
                self.memory.push(state,action,next_state,reward)
                state = next_state
                self.optimize_model()
                episode += 1
                if(episode % config.TARGET_UPDATE ==0 ):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                print(env.wealth,action,env.position)
            if (epoch + 1) % (interval) == 0 or epoch + 1 == num_epochs:
                torch.save(self.policy_net.state_dict(), 'models/DQN' + str(epoch + 1) + '.model')
    def test(self,num_epochs):
        env = Environment(test = True)
        state = (env.x[env.current_step], env.y_seq[env.current_step], env.position)
        while(1):
            action = self.select_action(state,test = True)
            _,next_state,_ = env.step(action.item())
            if(next_state == None):
                break
            state = next_state
            print(env.wealth)
# def to_variable(self, x):
#     if torch.cuda.is_available():
#         return Variable(torch.from_numpy(x).float()).cuda()
#     else:
#         return Variable(torch.from_numpy(x).float())
if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = config.BATCH_SIZE
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    agent = Agent(config.TIME_STEP,config.SPLIT_RATIO,lr)
    if not test:
        agent.train(num_epochs,interval)
    else:
        agent.load_model(DQN_path='models/DQN7.model')
        agent.test(num_epochs)