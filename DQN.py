import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name())


class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, n_actions)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, lr, state_dim, n_actions):

        self.dqn = DQN(state_dim, n_actions)
        self.dqn_target = DQN(state_dim, n_actions)
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.tensor(state).to(device)
        actions = self.dqn(state)
        return torch.argmax(actions).item()

    def learn(self, replay_buffer, gamma, batch_size, tau):

        # get sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).reshape(len(action), 1).to(device)
        reward = torch.FloatTensor(reward).reshape(len(reward), 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape(len(done), 1).to(device)

        # get target q value
        q_next = self.dqn_target(next_state).detach().max(1)[0].unsqueeze(1)
        q_target = reward + gamma * (1 - done) * q_next

        # calculate loss
        q_value = self.dqn(state)
        q_value = q_value.gather(1, action)
        q_loss = F.mse_loss(q_value, q_target)
        self.dqn_optimizer.zero_grad()
        q_loss.backward()
        self.dqn_optimizer.step()

        # soft update
        for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            target_param.data.copy_(((1 - tau) * target_param.data) + (tau * param.data))
