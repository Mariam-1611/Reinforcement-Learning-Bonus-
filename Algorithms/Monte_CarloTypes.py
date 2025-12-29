import random
import numpy as np

# Monte Carlo Every-Visit 
def monte_carlo_every_visit(enviro , episodes = 10000 , gamma = 0.9, stochastic=False, trace=False):
  returns = {}
  G = {}
  N = {}
  history = []
  
  for s in enviro.states:
    for a in enviro.actions:
      sa = (s , a)
      G[sa] = 0.0
      N[sa] = 0
      returns[sa] = []
  
  for episode in range(episodes):
    states , actions , rewards = generate_episode(enviro, stochastic=stochastic)
    if trace:
      history.append(sum(rewards))

    G_episode = 0.0
    for t in reversed(range(len(rewards))):
      state = states[t]
      action = actions[t]
      reward = rewards[t] 
      sa_pair = (state , action)
      G_episode = reward + gamma * G_episode
      returns[sa_pair].append(G_episode)
      N[sa_pair] += 1
      G[sa_pair] = sum(returns[sa_pair]) / N[sa_pair]
  
  Q = {}
  for s in enviro.states:
    Q[s] = {}
    for a in enviro.actions:
      Q[s][a] = G[(s,a)]
  
  if trace:
    return Q, history
  return Q

# Monte Carlo First-Visit 
def monte_carlo_first_visit(enviro , episodes = 10000 , gamma = 0.9, stochastic=False, trace=False):
  returns = {}
  G = {}
  N = {}
  history = []
  
  for s in enviro.states:
    for a in enviro.actions:
      sa = (s , a)
      G[sa] = 0.0
      N[sa] = 0
      returns[sa] = []
  
  for episode in range(episodes):
    states , actions , rewards = generate_episode(enviro, stochastic=stochastic)
    if trace:
      history.append(sum(rewards)) 
    G_episode = 0.0
    visited_state_action = set()
    for t in reversed(range(len(rewards))):
      state = states[t]
      action = actions[t]
      reward = rewards[t]
      sa_pair = (state , action)
      G_episode = reward + gamma * G_episode
      
      if sa_pair not in visited_state_action:
        returns[sa_pair].append(G_episode)
        N[sa_pair] += 1
        G[sa_pair] = sum(returns[sa_pair]) / N[sa_pair]
        visited_state_action.add(sa_pair)
  
  Q = {}
  for s in enviro.states:
    Q[s] = {}
    for a in enviro.actions:
      Q[s][a] = G[(s,a)]
  
  if trace:
    return Q, history
  return Q

def generate_episode(enviro , policy=None, stochastic=False, max_steps=500):
  states = []
  actions = []
  rewards = []
  state = (0,0)
  done = False
  steps = 0
  
  while not done and steps < max_steps:
    states.append(state)
    if policy is None:
      action = random.choice(enviro.actions)
    else:
      action = policy.get(state, random.choice(enviro.actions))
    actions.append(action)
    next_state , reward , done = enviro.transition(state , action, stochastic=stochastic)
    rewards.append(reward)
    state = next_state
    steps += 1
  return states , actions , rewards
