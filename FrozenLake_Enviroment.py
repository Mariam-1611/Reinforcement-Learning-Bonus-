from GridWorld_Enviroment import GridWorldEnvironment
import random
class FrozenLakEnviroment(GridWorldEnvironment):
    def __init__(self , rows=5 ,cols=5 ,slip = 0.2):
        super().__init__(rows,cols)
        self.slip = slip
        self.goal = (rows-1,cols-1)
        self.generat_holes()
        self.get_prob()

    def generat_holes(self):
        grids = []
        for s in self.states:
            if s not in {(0,0),self.goal}:
                grids.append(s)
        num_holes = min(int(0.2 * len(grids)), max(3, len(grids)//5))
        self.holes = random.sample(grids,num_holes)
        print(f"Generated {len(self.holes)} holes for {self.rows}x{self.cols}")

    def get_prob(self):
        self.Prob = {}
        for state in self.states:
            self.Prob[state] = {}
            for action in self.actions:
                self.Prob[state][action] = self.stochastic_transition_matrix(state,action)  

    def stochastic_transition_matrix(self,state,action):
        i , j = state
        if state == self.goal:
            return [(1.0, state, 10.0)]
        if action == "up":
            i_2 , j_2 = i - 1 ,j
        elif action == "down":
            i_2 , j_2 = i + 1 ,j
        elif action == "left":
            i_2 , j_2 = i ,j-1
        elif action == "right":
            i_2 , j_2 = i , j+1

        check_state , check_reward = self.reach_end(i_2,j_2,state)
        if action in ["up", "down"]: 
            slip1_i, slip1_j = i, j-1
            slip2_i, slip2_j = i, j+1
        else:  
            slip1_i, slip1_j = i-1, j
            slip2_i, slip2_j = i+1, j
    
        slip1_state, slip1_reward = self.reach_end(slip1_i, slip1_j, state)
        slip2_state, slip2_reward = self.reach_end(slip2_i, slip2_j, state)
    
        return [
            (1-self.slip, check_state, check_reward),
            (self.slip/2, slip1_state, slip1_reward),
            (self.slip/2, slip2_state, slip2_reward)
        ]

    def reach_end(self, i, j, state):
        next_state = (i,j)
        reward = 0
        if i < 0 or i >= self.rows or j < 0 or j >=self.cols:
            return state , -1
        if next_state == self.goal:
            reward = 10
        elif next_state in self.holes:
            reward = -1
        return next_state , reward 

    def transition(self, state, action, stochastic=False):
        """Return (next_state, reward, done) sampling according to self.Prob when stochastic=True."""
        outcomes = self.Prob.get(state, {}).get(action, [(1.0, state, 0.0)])
        if stochastic and len(outcomes) > 1:
            r = random.random()
            acc = 0.0
            for prob, nxt, rew in outcomes:
                acc += prob
                if r <= acc:
                    done = (nxt == self.goal or nxt in self.holes)
                    return nxt, rew, done
            last = outcomes[-1]
            done = (last[1] == self.goal or last[1] in self.holes)
            return last[1], last[2], done
        else:
            nxt, rew = outcomes[0][1], outcomes[0][2]
            done = (nxt == self.goal or nxt in self.holes)
            return nxt, rew, done

    def print_map_data(self):
        grids = [['F' for _ in range(self.cols)] for _ in range(self.rows)]
        grids[0][0] = 'S'
        grids[self.rows-1][self.cols-1] = 'G'
        for i,j in self.holes: 
            grids[i][j] = 'H'
        return grids
        

