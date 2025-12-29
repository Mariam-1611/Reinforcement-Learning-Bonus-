from FrozenLake_Enviroment import FrozenLakEnviroment
from GridWorld_Enviroment import GridWorldEnvironment
from Algorithms.Policy_Iteration import Policy_Iteration , policy_imporovment , policy_evaluation
from Algorithms.Value_Iteration import Value_Iteration 


# env = GridWorldEnvironment(rows=3, cols=3)
# policy , v_optimal = Value_Iteration(env)
# print("V values:", v_optimal)
# print("Best policy:", policy)
# policy , v = Policy_Iteration(env , v)
# print("V values:", v)
# print("Best policy:", policy)

env = FrozenLakEnviroment(4, 4, slip=0.2)
# print(env.Prob[(1,1)]['right'])
# policy, values = Policy_Iteration(env)  
values, optimal_policy = Value_Iteration(env, theta=0.0001, MAX=1000, gamma=0.9)
print("Optimal Values (first 5 states):")
for i, state in enumerate(env.states[:5]):
    print(f"State {state}: {values[state]:.3f}")

print("\nOptimal Policy:")
for state in [(0,0), (0,1), (1,1), (2,2), env.goal]:
    print(f"State {state}: {optimal_policy[state]}")
