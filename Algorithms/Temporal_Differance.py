import random
# Temporal Difference (TD(0))
def temporal_difference(enviro , episodes = 10000 , alpha = 0.1 , gamma = 0.9 , epsilon = 0.1, trace=False):
  Q = {}
  history = []
  for s in enviro.states:
    Q[s] = {}
    for a in enviro.actions:
      Q[s][a] = 0.0
  
  for episode in range(episodes):
    state = (0,0)
    done = False
    ep_return = 0.0
    
    while not done:
      if random.random() < epsilon:
        action = random.choice(enviro.actions)
      else:
        action = max(enviro.actions , key=lambda a: Q[state][a])
      
      next_state , reward , done = enviro.transition(state , action)
      ep_return += reward
      max_next_value = max(Q[next_state].values())
      td_target = reward + gamma * max_next_value
      td_error = td_target - Q[state][action]
      Q[state][action] += alpha * td_error
      
      state = next_state
    if trace:
      history.append(ep_return)
  
  if trace:
    return Q, history
  return Q
