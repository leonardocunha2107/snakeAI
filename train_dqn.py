from .agent import *
import torch.optim as optim
from .game import SnakeGame
import torch
from itertools import count
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F
BATCH_SIZE=128
GAMMA = 0.999

def optimize_model(optimizer,agent,target_net,memory,device='cuda'):
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action).view(-1,1)
    reward_batch = torch.stack(batch.reward).view(-1,1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = agent.net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    ret=float(loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    for param in agent.net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return ret

def train(num_episodes,show_last=5):
    fig=plt.figure()
    TARGET_UPDATE = 10
    BOARD_SHAPE=(10,10)
    
    device='cuda'
    #env=SnakeGame(dim=BOARD_SHAPE)
    agent=DQNAgent(BOARD_SHAPE,n_actions=)
    target_net=DQN(BOARD_SHAPE[0],BOARD_SHAPE[1],4).to(device)
    target_net.load_state_dict(agent.net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(agent.net.parameters())
    memory = ReplayMemory(10000)
    episode_durations=[]
    
    fig = plt.figure()
    
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = env.get_board()
        total_reward=0
        ims=[]
        if show_last and num_episodes-i_episode<=show_last :
            ims.append([plt.imshow(env.render())])
        losses=[]

        for t in count():
            # Select and perform an action

            action = agent.select_action(state)
            _, reward, done, _ = env.step(action.item())
            total_reward+=reward
            reward = torch.tensor([reward],dtype=torch.int, device=device)
    
            # Observe new state

            if not done:
                next_state=env.get_board()
            else:
                next_state = None
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            losses.append(optimize_model(optimizer,agent,target_net,memory))
            if ims:
                ims.append([plt.imshow(env.render())])

            if done:
                episode_durations.append(t + 1)
                #plot_durations()
                break
        if ims:
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                            repeat_delay=1000)
            plt.show()
        avg_loss=sum(losses)/len(losses)
        print(f"Finished episode {i_episode} with {episode_durations[-1]} steps \
               reward {total_reward} and average loss {avg_loss}")

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(agent.net.state_dict())
    torch.save(agent.net.state_dict(),'model.mdl')
    
if __name__=='__main__':
    train(50)