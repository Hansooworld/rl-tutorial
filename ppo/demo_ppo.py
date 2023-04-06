import torch, gym, tqdm, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from itertools import count

def np2torch(x: np.ndarray,
             dtype: torch.dtype = torch.float32,
             device: torch.device = torch.device('cpu')):
    return torch.from_numpy(x).type(dtype).to(device)

def build_mlp(in_dim: int,
              h_dims: list,
              h_actv: nn.Module):
    layers = []
    for i in range(len(h_dims)):
        if i == 0:
            layers.append(nn.Linear(in_dim, h_dims[i]))
        else:
            layers.append(nn.Linear(h_dims[i-1], h_dims[i]))
        layers.append(h_actv)
    return nn.Sequential(*layers)

def calc_gae(reward_batch: torch.Tensor,
             val_batch: torch.Tensor,
             done,
             last_val,
             gamma: float,
             lmbda: float):
    advantage_batch = np.zeros(shape=(reward_batch.shape[0], 1), dtype=np.float32)
    
    advantage = 0.0
    for idx in reversed(range(reward_batch.shape[0])):
        if idx == reward_batch.shape[0]-1:
            next_done = 1 - done
            next_val = last_val
        else:
            next_done = 1
            next_val = val_batch[idx+1]
        delta = reward_batch[idx] + gamma * next_val * next_done - val_batch[idx]
        advantage = delta + gamma * lmbda * next_done * advantage
        advantage_batch[idx] = advantage
    return_batch = advantage_batch + val_batch
    return return_batch, advantage_batch

class ActorClass(nn.Module):
    def __init__(self,
                 max_torque: float,
                 obs_dim: int,
                 h_dims: list,
                 act_dim: int,
                 h_actv: nn.Module,
                 mu_actv: nn.Module,
                 lr_actor: float,
                 ):
        super(ActorClass, self).__init__()
        self.max_torque = max_torque
        self.layers = build_mlp(in_dim=obs_dim, h_dims=h_dims, h_actv=h_actv)
        self.mu_head = nn.Linear(h_dims[-1], act_dim)
        self.mu_actv = mu_actv
        # self.apply(self._init_weights)

    def forward(self,
                obs: torch.Tensor):
        x = self.layers(obs)
        if self.mu_actv is not None:
            mu = self.mu_actv(self.mu_head(x))
        else:
            mu = self.mu_head(x)
            mu = (mu+1)*self.max_torque - self.max_torque
        return mu

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
class CriticClass(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 h_dims: list,
                 val_dim: int,
                 h_actv: nn.Module,
                 out_actv: nn.Module,
                 lr_critic: float):
        super(CriticClass, self).__init__()
        self.layers = build_mlp(in_dim=obs_dim, h_dims=h_dims, h_actv=h_actv)
        self.val_head = nn.Linear(h_dims[-1], val_dim)
        self.out_actv = out_actv
        if self.out_actv is not None:
            self.out_actv = out_actv
        # self.apply(self._init_weights)
    
    def forward(self,
                obs):
        x = self.layers(obs)
        val = self.val_head(x)
        if self.out_actv is not None:
            val = self.out_actv(val)
        return val

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

class PPOClass(nn.Module):
    def __init__(self,
                 max_torque: float,
                 obs_dim: int,
                 act_dim: int,
                 h_dims: list,
                 gamma: float,
                 lmbda: float,
                 lr_actorcritic: float,
                 clip_ratio: float,
                 value_coef: float,
                 entropy_coef: float,
                 max_grad: float
                 ):
        super(PPOClass, self).__init__()
        
        self.max_torque = max_torque
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef   
        self.entropy_coef = entropy_coef
        self.max_grad = max_grad
        
        self.actor = ActorClass(max_torque=max_torque,obs_dim=obs_dim,h_dims=h_dims,act_dim=act_dim,h_actv=nn.ReLU(),mu_actv=None,lr_actor=lr_actorcritic)
        self.critic = CriticClass(obs_dim=obs_dim,h_dims=h_dims,val_dim=1,h_actv=nn.ReLU(),out_actv=None,lr_critic=lr_actorcritic)
        self.log_std = nn.Parameter(torch.ones(act_dim) * torch.log(torch.tensor((1.0))), requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actorcritic)
    
    def forward(self,
                obs: torch.Tensor):
        mu = self.actor(obs)
        val = self.critic(obs)
        dist = Normal(mu, torch.exp(self.log_std))
        return dist, val
    
    def get_action(self,
                   obs):
        obs_torch = torch.unsqueeze(torch.FloatTensor(obs), 0)
        dist, val = self.forward(obs_torch)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=1)
        return action[0].detach().numpy(), torch.squeeze(log_prob).detach().numpy(), torch.squeeze(val).detach().numpy()
        
    def get_val(self,
                  obs):
        obs_torch = torch.unsqueeze(torch.FloatTensor(obs), 0)
        dist, val = self.forward(obs_torch)
        return torch.squeeze(val).detach().numpy()

    def eval_action(self,
                    obs_batch,
                    act_batch):
        obs_torch = obs_batch.clone().detach()
        action_torch = act_batch.clone().detach()
        dist, val = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return log_prob, val
    
    def update(self,
               obs_batch,
               act_batch,
               log_prob_batch,
               advantage_batch,
               return_batch):
        new_log_prob_batch, val_batch = self.eval_action(obs_batch, act_batch)
        ratio = torch.exp(new_log_prob_batch - log_prob_batch)
        
        surr1 = ratio * advantage_batch
        surr2 = torch.clip(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * advantage_batch
        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = self.value_coef * torch.mean((val_batch-return_batch)**2)        
        total_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad)
        self.optimizer.step()

        return actor_loss.detach(), critic_loss.detach(), total_loss.detach()

class PPOBufferClass:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 buffer_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size
        
        self.obs_buffer = np.zeros(shape=(buffer_size, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros(shape=(buffer_size, act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.log_prob_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.return_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.advantage_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.val_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)

        self.random_generator = np.random.default_rng()
        self.start_idx, self.pointer = 0, 0
        
    def put(self,
            obs,
            act,
            reward,
            val,
            log_prob):
        self.obs_buffer[self.pointer] = obs
        self.act_buffer[self.pointer] = act
        self.reward_buffer[self.pointer] = reward
        self.val_buffer[self.pointer] = val
        self.log_prob_buffer[self.pointer] = log_prob
        self.pointer += 1
    
    def get_gae_batch(self,
                      gamma,
                      lmbda,
                      done,
                      last_val):
        path_slice = slice(self.start_idx, self.pointer)
        val_mini_buffer = self.val_buffer[path_slice]
        
        self.return_buffer[path_slice], self.advantage_buffer[path_slice] = calc_gae(
                                                                                    self.reward_buffer[path_slice],
                                                                                    val_mini_buffer,
                                                                                    done,
                                                                                    last_val,
                                                                                    gamma,
                                                                                    lmbda
                                                                                    )
        self.start_idx = self.pointer
    
    def get_mini_batch(self,
                       mini_batch_size):
        assert mini_batch_size <= self.pointer
        indices = np.arange(self.pointer)
        self.random_generator.shuffle(indices)
        
        split_indices = []
        point = mini_batch_size
        while point < self.pointer:
            split_indices.append(point)
            point += mini_batch_size
        
        temp_data = {
                    'obs': np.split(self.obs_buffer[indices], split_indices),
                    'act': np.split(self.act_buffer[indices], split_indices),
                    'reward': np.split(self.reward_buffer[indices], split_indices),
                    'val': np.split(self.val_buffer[indices], split_indices),
                    'log_prob': np.split(self.log_prob_buffer[indices], split_indices),
                    'return': np.split(self.return_buffer[indices], split_indices),
                    'advantage': np.split(self.advantage_buffer[indices], split_indices)
                    }
        
        data = []
        for k in range(len(temp_data['obs'])):
            data.append({
                        'obs': temp_data['obs'][k],
                        'action': temp_data['act'][k],
                        'reward': temp_data['reward'][k],
                        'val': temp_data['val'][k],
                        'log_prob': temp_data['log_prob'][k],
                        'return': temp_data['return'][k],
                        'advantage': temp_data['advantage'][k]
                        })
        return data

    def clear(self):
        self.start_idx, self.pointer = 0, 0
    
def main():
    env = gym.make('Pendulum-v1')
    max_epi = 2000
    max_step = 200
    buffer_size = 2048
    mini_batch_size = 64
    n_step_per_update=buffer_size
    k_epoch = 10
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    h_dims = [128, 128]
    gamma = 0.99
    lmbda = 0.95
    lr_actorcritic = 3e-4
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad = 0.5
    
    PPO = PPOClass(max_torque=2.0,
                   obs_dim=obs_dim,
                   act_dim=act_dim,
                   h_dims=h_dims,
                   gamma=gamma,
                   lmbda=lmbda,
                   lr_actorcritic=lr_actorcritic,
                   clip_ratio=clip_ratio,
                   value_coef=value_coef,
                   entropy_coef=entropy_coef,
                   max_grad=max_grad)
    
    PPOBuffer = PPOBufferClass(obs_dim=obs_dim,
                               act_dim=act_dim,
                               buffer_size=buffer_size)
    epi_reward = 0
    epi_cnt = 0
    iter_cnt = 0
    actor_loss_ls = []
    critic_loss_ls = []
    total_loss_ls = []
    mean_rewards = []

    obs = env.reset()
    for t in range(1, 2048 * 500+1):
        action, log_prob, val = PPO.get_action(obs)
        real_action = np.clip(action, -PPO.max_torque, PPO.max_torque)
        next_obs, reward, done, info = env.step(real_action)
        
        epi_reward += reward

        PPOBuffer.put(obs, action, reward, val, log_prob)
        obs = next_obs

        if t % n_step_per_update == 0 or done:
            if done:
                epi_cnt += 1
            last_val = PPO.get_val(obs)
            PPOBuffer.get_gae_batch(gamma=PPO.gamma,
                                    lmbda=PPO.lmbda,
                                    done=done,
                                    last_val=last_val)
            obs = env.reset()
        
        if t % n_step_per_update == 0:
            iter_cnt += 1
            for _ in range(k_epoch):
                mini_batch_data = PPOBuffer.get_mini_batch(mini_batch_size=mini_batch_size)
                n_mini_batch = len(mini_batch_data)
                for k in range(n_mini_batch):
                    obs_batch = mini_batch_data[k]['obs']
                    act_batch = mini_batch_data[k]['action']
                    log_prob_batch = mini_batch_data[k]['log_prob']
                    advantage_batch = mini_batch_data[k]['advantage']
                    advantage_batch = (advantage_batch - np.squeeze(np.mean(advantage_batch, axis=0))) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)
                    return_batch = mini_batch_data[k]['return']
                    
                    obs_batch = np2torch(obs_batch)
                    act_batch = np2torch(act_batch)
                    log_prob_batch = np2torch(log_prob_batch)
                    advantage_batch = np2torch(advantage_batch)
                    return_batch = np2torch(return_batch)

                    actor_loss, critic_loss, total_loss = PPO.update(obs_batch, act_batch, log_prob_batch, advantage_batch, return_batch)
                    actor_loss_ls.append(actor_loss.numpy())
                    critic_loss_ls.append(critic_loss.numpy())
                    total_loss_ls.append(total_loss.numpy())
            PPOBuffer.clear()      
            
            mean_ep_reward = epi_reward / epi_cnt
            epi_reward, epi_cnt = 0, 0
            print(f"Season={iter_cnt} --> mean_ep_reward={mean_ep_reward}, pi_loss={np.mean(actor_loss_ls)}, v_loss={np.mean(critic_loss_ls)}, total_loss={np.mean(total_loss_ls)}")
            
            if iter_cnt % 10 == 0:
                if not os.path.exists("./model"):
                    os.makedirs("./model")
                torch.save(PPO.actor.state_dict(), f"./model/actor_{iter_cnt}.pth")
                torch.save(PPO.critic.state_dict(), f"./model/critic_{iter_cnt}.pth")
                    
if __name__ == "__main__":
    main()

    