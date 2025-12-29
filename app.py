from flask import Flask, render_template, jsonify, request
import time
import random
from FrozenLake_Enviroment import FrozenLakEnviroment
from GridWorld_Enviroment import GridWorldEnvironment
from Algorithms.Policy_Iteration import Policy_Iteration , policy_imporovment , policy_evaluation
from Algorithms.Value_Iteration import Value_Iteration
from Algorithms.Monte_CarloTypes import monte_carlo_every_visit, monte_carlo_first_visit
from Algorithms.Temporal_Differance import temporal_difference
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/env/<env_name>')
def env_detail(env_name):
    return render_template('env_detail.html', env_name=env_name)

@app.route('/api/run_algorithm')
def run_algorithm():
    try:
        rows = int(request.args.get('rows', 4))
        cols = int(request.args.get('cols', 4))
        slip = float(request.args.get('slip', 0.2))
        gamma = float(request.args.get('gamma', 0.9))
        theta = float(request.args.get('theta', 0.0001))
        
        algorithm = request.args.get('algorithm')
        env_name = request.args.get('env', 'frozenlake')

        if env_name == 'frozenlake':
            env = FrozenLakEnviroment(rows, cols, slip)
        elif env_name == 'gridworld':
            env = GridWorldEnvironment(rows, cols)
        else:
            return jsonify({'error': 'Unknown environment'}), 400

        trace = request.args.get('trace', '0').lower() in ('1','true','yes')

        start_time = time.time()
        if algorithm == 'policy_iteration':
            if trace:
                policy, values, iterations, history = Policy_Iteration(env, theta=theta, gamma=gamma, trace=True)
            else:
                policy, values, iterations = Policy_Iteration(env, theta=theta, gamma=gamma)
            result_type = 'policy_iteration'
        elif algorithm == 'value_iteration':
            if trace:
                values, policy, iterations, history = Value_Iteration(env, theta=theta, gamma=gamma, trace=True)
            else:
                values, policy, iterations = Value_Iteration(env, theta=theta, gamma=gamma)
            result_type = 'value_iteration'
        elif algorithm in ('td','temporal_difference'):
            episodes = int(request.args.get('episodes', 2000))
            alpha = float(request.args.get('alpha', 0.1))
            epsilon = float(request.args.get('epsilon', 0.1))
            if trace:
                Q, history = temporal_difference(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, trace=True)
                result_episode_returns = history
            else:
                Q = temporal_difference(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
                result_episode_returns = None
            policy = {}
            values = {}
            for s in env.states:
                best_a = None
                best_v = float('-inf')
                if isinstance(Q.get(s), dict):
                    for a in env.actions:
                        v = Q[s].get(a, 0)
                        if v > best_v:
                            best_v = v
                            best_a = a
                else:
                    best_a = env.actions[0]
                    best_v = 0
                policy[s] = best_a
                values[s] = best_v
            iterations = episodes
            result_type = 'td'
        elif algorithm in ('mc_every','mc_first'):
            episodes = int(request.args.get('episodes', 2000))
            stochastic_mc = request.args.get('stochastic', '0').lower() in ('1','true','yes')
            if trace:
                if algorithm == 'mc_every':
                    Q, history = monte_carlo_every_visit(env, episodes=episodes, gamma=gamma, stochastic=stochastic_mc, trace=True)
                else:
                    Q, history = monte_carlo_first_visit(env, episodes=episodes, gamma=gamma, stochastic=stochastic_mc, trace=True)
                result_episode_returns = history
            else:
                if algorithm == 'mc_every':
                    Q = monte_carlo_every_visit(env, episodes=episodes, gamma=gamma, stochastic=stochastic_mc)
                else:
                    Q = monte_carlo_first_visit(env, episodes=episodes, gamma=gamma, stochastic=stochastic_mc)
                result_episode_returns = None
            policy = {}
            values = {}
            for s in env.states:
                best_a = None
                best_v = float('-inf')
                if isinstance(Q.get(s), dict):
                    for a in env.actions:
                        v = Q[s].get(a, 0)
                        if v > best_v:
                            best_v = v
                            best_a = a
                else:
                    best_a = env.actions[0]
                    best_v = 0
                policy[s] = best_a
                values[s] = best_v
            iterations = episodes
            result_type = algorithm
        else:
            return jsonify({'error': 'Unknown algorithm'}), 400
        time_taken = time.time() - start_time
        policy_serial = {f"({s[0]},{s[1]})": policy[s] for s in policy}
        values_serial = {f"({s[0]},{s[1]})": float(values[s]) for s in values}

        result = {
            'policy': policy_serial,
            'values': values_serial,
            'type': result_type,
            'iterations': iterations,
            'time_taken': time_taken,
            'env': env_name
        }

        # include rewards
        try:
            rewards_map = env.reward_map()
            rewards_serial = {f"({s[0]},{s[1]})": float(rewards_map[s]) for s in rewards_map}
            rewards_grid = env.reward_grid()
            result['rewards'] = rewards_serial
            result['rewards_grid'] = rewards_grid
        except Exception:
            result['rewards'] = {}
            result['rewards_grid'] = []

        if trace:
            result['history'] = history
            if 'result_episode_returns' in locals() and result_episode_returns is not None:
                result['episode_returns'] = result_episode_returns

        simulate_flag = request.args.get('simulate', '0').lower() in ('1','true','yes')
        stochastic_sim = request.args.get('stochastic', '0').lower() in ('1','true','yes')
        if simulate_flag:
            try:
                def sample_outcome(outcomes):
                    r = random.random()
                    acc = 0.0
                    for prob, next_state, reward in outcomes:
                        acc += prob
                        if r <= acc:
                            return next_state, reward
                    return outcomes[-1][1], outcomes[-1][2]

                def simulate_policy(env, policy, max_steps=200, stochastic=False):
                    traj = []
                    state = (0,0)
                    cum = 0.0
                    for step in range(max_steps):
                        if state == env.goal:
                            traj.append({'step': step, 'state': f"({state[0]},{state[1]})", 'reward': 0.0, 'cum_reward': cum})
                            break
                        action = policy.get(state)
                        if action is None:
                            break
                        outcomes = env.Prob[state][action]
                        if stochastic:
                            next_state, reward = sample_outcome(outcomes)
                        else:
                            next_state, reward = outcomes[0][1], outcomes[0][2]
                        cum += float(reward)
                        traj.append({'step': step, 'state': f"({state[0]},{state[1]})", 'next_state': f"({next_state[0]},{next_state[1]})", 'action': action, 'reward': float(reward), 'cum_reward': cum})
                        state = next_state
                        if state == env.goal:
                            traj.append({'step': step+1, 'state': f"({state[0]},{state[1]})", 'reward': 0.0, 'cum_reward': cum})
                            break
                    return traj

                policy_for_sim = {s: policy[s] for s in policy}
                trajectory = simulate_policy(env, policy_for_sim, stochastic=stochastic_sim)
                result['trajectory'] = trajectory
            except Exception:
                result['trajectory'] = []


        # Map visualization data
        result['map'] = env.print_map_data()
        result['holes'] = env.holes
        result['rows'] = rows
        result['cols'] = cols
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulate_policy', methods=['POST'])
def simulate_policy_endpoint():
    data = request.get_json(force=True)
    env_name = data.get('env', 'frozenlake')
    rows = int(data.get('rows', 4))
    cols = int(data.get('cols', 4))
    slip = float(data.get('slip', 0.0))
    stochastic = bool(data.get('stochastic', False))
    policy = data.get('policy', {})
    try:
        if env_name == 'frozenlake':
            env = FrozenLakEnviroment(rows, cols, slip)
        elif env_name == 'gridworld':
            env = GridWorldEnvironment(rows, cols)
        else:
            return jsonify({'error': 'Unknown environment'}), 400
        policy_tup = {}
        for k, v in policy.items():
            try:
                nums = k.replace('(', '').replace(')', '').split(',')
                policy_tup[(int(nums[0]), int(nums[1]))] = v
            except Exception:
                continue
        def sample_outcome(outcomes):
            r = random.random()
            acc = 0.0
            for prob, next_state, reward in outcomes:
                acc += prob
                if r <= acc:
                    return next_state, reward
            return outcomes[-1][1], outcomes[-1][2]

        def simulate_policy_local(env, policy, max_steps=200, stochastic=False):
            traj = []
            state = (0,0)
            cum = 0.0
            for step in range(max_steps):
                if state == env.goal:
                    traj.append({'step': step, 'state': f"({state[0]},{state[1]})", 'reward': 0.0, 'cum_reward': cum})
                    break
                action = policy.get(state)
                if action is None:
                    break
                outcomes = env.Prob[state][action]
                if stochastic:
                    next_state, reward = sample_outcome(outcomes)
                else:
                    next_state, reward = outcomes[0][1], outcomes[0][2]
                cum += float(reward)
                traj.append({'step': step, 'state': f"({state[0]},{state[1]})", 'next_state': f"({next_state[0]},{next_state[1]})", 'action': action, 'reward': float(reward), 'cum_reward': cum})
                state = next_state
                if state == env.goal:
                    traj.append({'step': step+1, 'state': f"({state[0]},{state[1]})", 'reward': 0.0, 'cum_reward': cum})
                    break
            return traj

        trajectory = simulate_policy_local(env, policy_tup, stochastic=stochastic)
        return jsonify({'trajectory': trajectory, 'map': env.print_map_data(), 'rows': rows, 'cols': cols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
