import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it 
import seaborn as sns

class MDP:
    def __init__(self, transition_function, reward_function):
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.states = list(set([s for s, _, _ in transition_function.keys()] + [s_prime for _, _, s_prime in transition_function.keys()]))
        self.actions = list(set([a for _, a, _ in transition_function.keys()]))
        
        # Set reward to 0 for each item in the transition function not already in the reward function
        for key in transition_function.keys():
            if key not in reward_function:
                reward_function[key] = 0
        
        # Add transitions for states with only one action
        for state in self.states:
            actions = [a for (s, a, _) in transition_function.keys() if s == state]
            if len(actions) <= 1:
                for action in self.actions:
                    if action not in actions:
                        transition_function[(state, action, state)] = 1.0
                        reward_function[(state, action, state)] = 0
        
        # Add zero-probability transitions for non-terminal states
        for state in self.states:
            for action in self.actions:
                next_states = [s_prime for (s, a, s_prime) in transition_function.keys() if s == state and a == action]
                for s_prime in self.states:
                    if s_prime not in next_states:
                        transition_function[(state, action, s_prime)] = 0.0
                        reward_function[(state, action, s_prime)] = 0

    def next_state_and_reward(self, state, action):
        next_states = [s_prime for (s, a, s_prime) in self.transition_function.keys() if s == state and a == action]
        probabilities = [self.transition_function[(state, action, s_prime)] for s_prime in next_states]
        next_state = np.random.choice(next_states, p=probabilities)
        reward = self.reward_function[(state, action, next_state)]
        return next_state, reward

    def is_terminal(self, state):
        return all(self.transition_function[(state, action, state)] == 1.0 for action in self.actions)

    def visualize(self):
        G = nx.MultiDiGraph()
        for (s, a, s_prime), prob in self.transition_function.items():
            if s != s_prime:
                if prob > 0:  # Only add edges with probability > 0
                    G.add_edge(s, s_prime, label=f'{a} P={prob} R={self.reward_function[(s, a, s_prime)]}')
        
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        
        # Jitter the positions of the edges
        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
        
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=10)
        nx.draw_networkx_edges(G, pos, edge_color="grey", connectionstyle=connectionstyle)
        
        labels = {
            tuple(edge): f"{attrs['label']}"
            for *edge, attrs in G.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=labels,
            connectionstyle=connectionstyle,
            label_pos=0.3,
            font_color="yellow",
            bbox={"alpha": 1},
        )

        plt.title('MDP Visualization')
        plt.show()

def get_next_states(mdp, state, action):
    if mdp.is_terminal(state):
        next_states=['terminal_state']
    else:
        next_states = [s_prime for (s, a, s_prime) in mdp.transition_function.keys() if s == state and a == action]
    return next_states

def planning(s,a, mdp, gamma, cost):
    q_value=0
    next_states=get_next_states(mdp,s,a)
    for next_state in next_states:
        if next_state=='terminal_state':
            q_value=0
            return q_value
        else:
            q_value += mdp.transition_function[(s, a, next_state)] * (mdp.reward_function[(s, a, next_state)] + gamma*max(planning(next_state,next_a, mdp, gamma, cost) for next_a in mdp.actions)) 
            return q_value
def get_max_index(lst):
    return lst.index(max(lst)) if lst else None

def agent_act(mdp, policy, S0, strategy, num_episodes, num_steps, learning_function=None, learning_rate=0.1, inverse_temp=1.0, gamma=0.9, cost=0.1):
    state_occupancies = {state: 0 for state in mdp.states}
    total_rewards = []
    Q = {(state, action): 0 for state in mdp.states for action in mdp.actions}

    for episode in range(num_episodes):
        state = S0
        total_reward = 0
        for step in range(num_steps):
      
            if mdp.is_terminal(state):
                break
            if strategy == 'learn':
                action_probs = np.exp([Q[(state, action)] * inverse_temp for action in mdp.actions])
                action_probs /= np.sum(action_probs)
                action = np.random.choice(mdp.actions, p=action_probs)
            elif strategy == 'static':
                action = max(policy[state], key=policy[state].get)
            elif strategy == 'plan':
                planning_q=[]
                a_order=[]
                for a in mdp.actions: 
                    a_order.append(a)
                    planning_q.append(planning(state,a, mdp, gamma, cost))

                a_index=get_max_index(planning_q)
                action=a_order[a_index]
            
            next_state, reward = mdp.next_state_and_reward(state, action)
           

            state_occupancies[next_state] += 1
            total_reward += reward
            if strategy == 'learn' and learning_function:
                Q = learning_function(Q, state, action, reward, next_state, learning_rate)
            state = next_state
        total_rewards.append(total_reward)

    return state_occupancies, total_rewards

def plot_average_reward(total_rewards, num_episodes):
    sns.barplot(y=total_rewards)
    plt.xlabel('Total_Reward')
    plt.title('Cumulative Reward Average')
    plt.show()

# Temporal Difference Learning function
def temporal_difference_learning(Q, state, action, reward, next_state, learning_rate, gamma=0.9):
    best_next_action = max(Q[(next_state, a)] for a in mdp.actions)
    Q[(state, action)] += learning_rate * (reward + gamma * best_next_action - Q[(state, action)])
    return Q

# Example usage
transition_function = {('s1', 'a1', 's2'): 0.8, ('s1', 'a1', 's3'): 0.2, ('s1', 'a2', 's2'): 0.2, ('s1', 'a2', 's3'): 0.8,
 ('s2', 'a2', 's4'): 0.8, ('s2', 'a2', 's5'): 0.2, ('s2', 'a1', 's5'): 0.8, ('s2', 'a1', 's6'): 0.2,
('s3', 'a2', 's5'): 0.8, ('s3', 'a2', 's6'): 0.2, ('s3', 'a1', 's6'): 0.8, ('s3', 'a1', 's5'): 0.2}
reward_function = {('s1', 'a2', 's4'): -500,('s1', 'a2', 's3'): -300,('s3', 'a1', 's6'): 600}
mdp = MDP(transition_function, reward_function)
policy = {'s1': {'a1': 0.5, 'a2': 0.5}, 's2': {'a1': 1.0},'s3':{'a2':1.0}}
S0 = 's1'
strategy = 'plan'
num_episodes = 1000
num_steps = 3
learning_rate = 0.1
inverse_temp = 1.0
gamma = 0.9
cost = 100
max_depth = 10

state_occupancies, total_rewards = agent_act(mdp, policy, S0, strategy, num_episodes, num_steps, temporal_difference_learning, learning_rate, inverse_temp, gamma, cost)

plot_average_reward(total_rewards, num_episodes)

# Visualize the MDP
# mdp.visualize()
