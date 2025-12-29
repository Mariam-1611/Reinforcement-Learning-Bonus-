from Algorithms.Policy_Iteration import policy_imporovment
def Value_Iteration(enviro , theta = 0.0001 , MAX = 1000 , gamma = 0.9, trace=False):
  v = {}
  for s in enviro.states:
    v[s] = 0.0

  iterations = 0
  history = []
  for i in range(MAX):
    iterations = i + 1
    delta = 0.0
    for s in enviro.states:
      old_value = v[s]
      max_value = float('-inf')
      for action in enviro.actions:
        intial_value = 0.0
        for probability , next_state , reward in enviro.Prob[s][action]:
          intial_value += probability *(reward + gamma *v[next_state]) 
        max_value = max(max_value , intial_value)
      v[s] = max_value
      delta = max(delta,abs(max_value - old_value))

    # collect trace
    if trace:
      values_serial = {f"({s[0]},{s[1]})": float(v[s]) for s in v}
      policy_serial = policy_imporovment(enviro, v)
      policy_serial = {f"({s[0]},{s[1]})": policy_serial[s] for s in policy_serial}
      history.append({'iteration': iterations, 'policy': policy_serial, 'values': values_serial})

    if delta < theta:
      break
  
  optimaPolicy = policy_imporovment(enviro , v)
  if trace:
    return v , optimaPolicy, iterations, history
  return v , optimaPolicy, iterations