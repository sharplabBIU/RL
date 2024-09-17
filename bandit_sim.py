import numpy as np
import matplotlib.pyplot as plt

# Define the bandit probabilities
bandit_probs = [0.8, 0.5, 0.2]

# Define the number of trials and episodes
n_trials = 50
n_episodes = 10000

# Define the learning rates for the agents
learning_rates = [0.9, 0.7, 0.5]

# Initialize the Q-values
initial_q = 0.5

# Initialize the Q-values
invtemp = 10

# Define the softmax function
def softmax(q_values, beta=invtemp):
    exp_q = np.exp(beta * q_values)
    return exp_q / np.sum(exp_q)

# Define the Rescorla-Wagner update rule
def rescorla_wagner(q, reward, alpha):
    return q + alpha * (reward - q)

# Initialize the agents
agents = [{'q_values': np.full(3, initial_q), 'alpha': alpha} for alpha in learning_rates]

# Add the 4th agent with dynamic learning rate
dynamic_agent = {'q_values': np.full(3, initial_q), 'alpha': np.ones(3), 'counts': np.zeros(3)}

# Store the rewards and choices
rewards = np.zeros((len(agents) + 1, n_episodes, n_trials))
choices = np.zeros((len(agents) + 1, n_episodes, n_trials), dtype=int)

# Run the episodes
for episode in range(n_episodes):
    # Reset the Q-values for each episode
    for agent in agents:
        agent['q_values'] = np.full(3, initial_q)
    dynamic_agent['q_values'] = np.full(3, initial_q)
    dynamic_agent['alpha'] = np.ones(3)
    dynamic_agent['counts'] = np.zeros(3)
    
    for t in range(n_trials):
        for i, agent in enumerate(agents):
            # Select an action using softmax
            action_probs = softmax(agent['q_values'])
            action = np.random.choice(len(bandit_probs), p=action_probs)
            
            # Get the reward
            reward = np.random.binomial(1, bandit_probs[action])
            
            # Update the Q-value
            agent['q_values'][action] = rescorla_wagner(agent['q_values'][action], reward, agent['alpha'])
            
            # Store the reward and choice
            rewards[i, episode, t] = reward
            choices[i, episode, t] = action
        
        # Dynamic agent
        action_probs = softmax(dynamic_agent['q_values'])
        action = np.random.choice(len(bandit_probs), p=action_probs)
        
        # Get the reward
        reward = np.random.binomial(1, bandit_probs[action])
        
        # Update the Q-value
        dynamic_agent['counts'][action] += 1
        dynamic_agent['alpha'][action] = 1 / dynamic_agent['counts'][action]
        dynamic_agent['q_values'][action] = rescorla_wagner(dynamic_agent['q_values'][action], reward, dynamic_agent['alpha'][action])
        
        # Store the reward and choice
        rewards[-1, episode, t] = reward
        choices[-1, episode, t] = action

# Calculate the average rewards over episodes
avg_rewards = np.mean(rewards, axis=1)


# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(4, 5))

# Plot the average reward over time
for i, agent in enumerate(agents):
    axs[0].plot(avg_rewards[i], label=f'Agent {i+1} (alpha={agent["alpha"]})')

# Plot the dynamic agent's average reward
axs[0].plot(avg_rewards[-1], label='Dynamic Agent')

axs[0].set_xlabel('Trial')
axs[0].set_ylabel('Average Reward')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Plot the choices over time
for i, agent in enumerate(agents):
    axs[1].plot(np.mean(choices[i]+1, axis=0), label=f'Agent {i+1} (alpha={agent["alpha"]})')

# Plot the dynamic agent's choices
axs[1].plot(np.mean(choices[-1]+1, axis=0), label='Dynamic Agent')

axs[1].set_xlabel('Trial')
axs[1].set_ylabel('Chosen Bandit')
axs[1].set_yticks([1, 2, 3])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Plot the latent reward values of each bandit
for i, prob in enumerate(bandit_probs):
    axs[2].plot([prob] * n_trials, label=f'Bandit {i+1} (p={prob})')

axs[2].set_xlabel('Trial')
axs[2].set_ylabel('Latent Reward Value')
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()