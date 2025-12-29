import random
class GridWorldEnvironment:
  def __init__(self, rows=5 , cols=5):
      self.rows = rows
      self.cols = cols
      self.states = []
      for i in range (rows):
        for j in range (cols):
          self.states.append((i,j))

      self.actions = ["up","down","left","right"]
      self.goal = (rows-1 , cols-1)
      self.holes = []
      self.Prob = {}
      for state in self.states:
        self.Prob[state] = {}
        for action in self.actions:
          i, j = state
          if state == self.goal:
            next_state, rewrad = state, 10
          else:
            if action == "up":
              i_2, j_2 = i - 1, j
            elif action == "down":
              i_2, j_2 = i + 1, j
            elif action == "left":
              i_2, j_2 = i, j - 1
            elif action == "right":
              i_2, j_2 = i, j + 1

            if i_2 < 0 or i_2 >= self.rows or j_2 < 0 or j_2 >= self.cols:
              next_state = state
              rewrad = -1
            else:
              next_state = (i_2, j_2)
              rewrad = 0
            if next_state == self.goal:
              rewrad = 10
          self.Prob[state][action] = [(1.0, next_state, rewrad)]

  def transition(self, state, action, stochastic=False):
    outcomes = self.Prob.get(state, {}).get(action, [(1.0, state, 0)])
    if stochastic and len(outcomes) > 1:
      r = random.random()
      acc = 0.0
      for prob, nxt, rew in outcomes:
        acc += prob
        if r <= acc:
          return nxt, rew, (nxt == self.goal)
      return outcomes[-1][1], outcomes[-1][2], (outcomes[-1][1] == self.goal)
    else:
      nxt, rew = outcomes[0][1], outcomes[0][2]
      done = (nxt == self.goal)
      return nxt, rew, done
  
  def print_map_data(self):
    grids = [['' for _ in range(self.cols)] for _ in range(self.rows)]
    grids[self.rows-1][self.cols-1] = 'G'
    return grids