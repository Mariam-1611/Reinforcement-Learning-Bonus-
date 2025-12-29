# Policy Evaluation 
def policy_evaluation(givenPolicy , enviro , theta = 0.0001 , MAX = 1000 , gamma = 0.9):
  v = {}
  for s in enviro.states:
    v[s] = 0.0
  
  for i in range(MAX):
    delta = 0.0
    for s in enviro.states:
      old_value = v[s]
      new_value = 0.0
      action = givenPolicy[s]
      for probability , next_state , reward in enviro.Prob[s][action]:
        new_value += probability *(reward + gamma * v[next_state])
      v[s] = new_value 
      delta = max(delta , abs(new_value - old_value))

    if delta < theta:
      break 
  return v

# Policy Improvment 
def policy_imporovment(enviro , v , gamma = 0.9):
  better_policy = {}
  for s in enviro.states:
    bestAction = None 
    bestValue = float('-inf')
    for action in enviro.actions:
      value = 0.0
      for probability , next_state , reward in enviro.Prob[s][action]:
        value += probability *(reward + gamma * v[next_state])

      if value > bestValue:
        bestValue = value
        bestAction = action 
    better_policy[s] =  bestAction
  return better_policy

# Policy Iteration
def Policy_Iteration(enviro , theta = 0.0001 , MAX = 1000 , gamma = 0.9, trace=False):
  policy = {}
  for state in enviro.states:
    policy[state] = enviro.actions[0]
  iterations = 0
  history = []
  while True:
    iterations += 1
    new_value = policy_evaluation(policy , enviro , theta , MAX , gamma)
    policy_imporved = policy_imporovment(enviro , new_value)  
    if trace:
      policy_serial = {f"({s[0]},{s[1]})": policy_imporved[s] for s in policy_imporved}
      values_serial = {f"({s[0]},{s[1]})": float(new_value[s]) for s in new_value}
      history.append({'iteration': iterations, 'policy': policy_serial, 'values': values_serial})

    if policy_imporved == policy:
      break
    policy = policy_imporved
  if trace:
    return policy, new_value, iterations, history
  return policy, new_value, iterations
