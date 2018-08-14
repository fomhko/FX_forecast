import config
import numpy as np
from dataset import Dataset
import torch
from replay_memory import ReplayMemory,Transition
from torch.autograd import Variable
class Environment:
    def __init__(self,init_position,test = False,unit = 0.01):
        self.dataset = Dataset(T = config.TIME_STEP,split_ratio = config.SPLIT_RATIO,binary_file = config.BINARY_DATASET)
        self.current_step = 0
        self.position = self.to_variable(init_position)#pos[0]: USDposition
        self.wealth = 10000;
        self.unit = unit
        if not test:
            self.x, self.y,self.y_seq = self.dataset.get_train_set()
            self.x = self.to_variable(self.x)
            self.y = self.to_variable(self.y)
            self.y_seq = self.to_variable(self.y_seq)
        else:
            self.x, self.y,self.y_seq = self.dataset.get_test_set()
            self.x = self.to_variable(self.x)
            self.y = self.to_variable(self.y)
            self.y_seq = self.to_variable(self.y_seq)
        self.horizon = self.x.shape[0]

    def reset(self):
        self.current_step = 0
    def step(self,action):#action 0 short 1 long 2 neutral
        prev_position = self.position.clone()
        cur_state = (self.x[self.current_step].unsqueeze(0),self.y_seq[self.current_step].unsqueeze(0),prev_position.unsqueeze(0))
        if(self.current_step+1 < self.horizon):
            if action == 0:
                self.position[0] = min(1,self.position[0] + self.unit)
                self.position[1] = max(0,self.position[1] - self.unit)
            elif action == 1:
                self.position[0] = max(0,self.position[0] - self.unit)
                self.position[1] = min(1,self.position[1] + self.unit)
            reward = (self.position[1]*(self.y[self.current_step] - self.y[self.current_step+1])/self.y[self.current_step+1])
            self.wealth = (reward+1) * self.wealth
            self.position[1] = (self.y[self.current_step]*self.position[1])/self.y[self.current_step+1]
            self.position[1] = self.position[1] / (self.position[1]+self.position[0])
            self.position[0] = 1-self.position[1]
            self.current_step += 1
            next_state = (self.x[self.current_step].unsqueeze(0),self.y_seq[self.current_step].unsqueeze(0),self.position.unsqueeze(0))
        else:
            reward = 0
            next_state = None
        return cur_state,next_state,self.to_variable(np.array([reward]))
    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

if __name__ == '__main__':
    env = Environment(test = False,init_position=np.array([0,1]))
    replay_memory = ReplayMemory(100)
    max_val = 0
    min_val = env.wealth
    while(1):
        action = np.random.randint(3)
        cur_state,next_state,reward= env.step(action)
        replay_memory.push(cur_state,action,next_state,reward)
        if(next_state == None):
            break
        max_val = max(max_val,env.wealth)
        min_val = min(min_val,env.wealth)
    print(max_val,min_val)

    # transitions = replay_memory.sample(1)
    # Transition(*zip(*transitions))
    # batch = Transition(*zip(*transitions))
    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)
    # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
