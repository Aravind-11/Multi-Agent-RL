import numpy as np
import gym

class Football:  # The class encapsulating the environment
    '''
    Actions [0 : Stand, 1 : Up, 2 : Right, 3 : Down, 4 : Left]
    These are the representing no.s for the mentioned actions
    '''

    def __init__(self, length=15, width=15, goalPositions=[5, 12]):
        
        # The player start at random locations
        
        self.pA=np.array([np.random.randint(length), np.random.randint(length)]) 
        self.pB=np.array([np.random.randint(length), np.random.randint(length)]) 
        
        self.h = length   # Length of the Football Pitch    
        self.w = width    # Width of the Football Pitch
        
        self.goalPositions = np.array(goalPositions)   # This means that the middle 4 positions at the right and left are the goals
        
        self.reward = np.array([0,0])
        
                                  # Initially the reward is 0
        
        self.observation_a=np.random.rand(5,)
        self.observation_b=np.random.rand(5,)
        self.done_a = bool(0)  
        self.done_b=bool(0)  
        self.observation_space=gym.spaces.Box(low=-8, high=8,
                                        shape=(5,), dtype=np.float32)
        self.ballOwner = np.random.randint(0,2)
        self.action_space=gym.spaces.Discrete(5)
    

    def isInBoard(self, x, y):
        if(x<0 or x>14):
          return 0
        if(y<0 or y>14):
          return 0 
        return 1
    
    def actionToMove(self, action):
        switcher = {
            0: [0, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [-1, 0],
        }
        return switcher.get(action)
class Agent_AB(Football,gym.Env):
  def __init__(self, length=15, width=15, goalPositions=[5, 12]):
    super().__init__()
    
    
  
  def step_a(self, action):
        if self.done_a == 1:
          self.reset_a()
        self.move_a(action)                   # We chose the first player at random !!! important thing to consider - how to choose first player . 
        if self.done_a == 1:
          return self.observation_a, self.reward[0], self.done_a,{}

        return self.observation_a, self.reward[0].astype(float), self.done_a,{}
  
  def move_a(self, action):
        
        newPosition = self.pA + self.actionToMove(action)

        # If it's opponent position
        if (newPosition == self.pB).any():
            self.ballOwner = 1
            self.reward[0]=-20
            self.reward[1]=20
        # If it's a goal
        if self.ballOwner is 0 and self.isInGoal_a(*newPosition) >= 0:
            self.done_a = 1
            return 1 - self.isInGoal_a(*newPosition)
        # If it's in the board
        if self.isInBoard(*newPosition):
          if(self.ballOwner is 0):
            self.reward[0] =  -0.1 * ((abs(newPosition[0]-14))+abs(newPosition[1]-8) )
          if(self.ballOwner is 1):
            self.reward[0] =  -0.1 * ((abs(newPosition[0]-self.pB[0]))+abs(newPosition[1]-self.pB[1]))
            self.pA = newPosition
        self.observation_a=np.array((*self.pA,*self.pB,self.ballOwner)).astype(np.float32)
        return -1
  def step_b(self, action):
        if self.done_b == 1:
          self.reset_b()
        self.move_b(action)                   # We chose the first player at random !!! important thing to consider - how to choose first player . 
        if self.done_b == 1:
          return self.observation_b, self.reward[1], self.done_b,{}

        return self.observation_b, self.reward[1].astype(np.float), self.done_b,{}
  
  def move_b(self, action):
        
        newPosition = self.pB + self.actionToMove(action)

        # If it's opponent position
        if (newPosition == self.pA).any():
            self.ballOwner = 0
            self.reward[1]=-20
            self.reward[0]=20
        # If it's a goal
        if self.ballOwner is 1 and self.isInGoal_b(*newPosition) >= 0:
            self.done_b = 1
            return 1 - self.isInGoal_b(*newPosition)
        # If it's in the board
        if self.isInBoard(*newPosition):
          if(self.ballOwner is 1):
            self.reward[0] =  -0.1 * ((abs(newPosition[0]-14))+abs(newPosition[1]-8) )
          if(self.ballOwner is 0):
            self.reward[0] =  -0.1 * ((abs(newPosition[0]-self.pA[0]))+abs(newPosition[1]-self.pA[1]))
            self.pB = newPosition
        self.observation_b=np.array((*self.pB,*self.pA,self.ballOwner)).astype(np.float32)
        return -1
  def reset_a(self):
        self.done_a = bool(0)
        self.reward = np.array([0,0])
        
        self.pA = np.array([np.random.randint(self.h), np.random.randint(self.h)])
        self.pB = np.array([np.random.randint(self.h), np.random.randint(self.h)]) 
        return np.array((*self.pA,*self.pB,self.ballOwner)).astype(np.float32)
  def reset_b(self):
        self.done_b = bool(0)
        
        #self.pA = np.array([np.random.randint(self.h), np.random.randint(self.h)])
        #self.pB = np.array([np.random.randint(self.h), np.random.randint(self.h)])
        return np.array((*self.pB,*self.pA,self.ballOwner)).astype(np.float32)
  def render(self,mode='console'):
        board = ''
        for y in range(self.h)[::-1]:
            for x in range(self.w):
                if ([x, y] == self.pA).all():
                    board += 'A' if self.ballOwner is 0 else 'a'
                elif ([x, y] == self.pB).all():
                    board += 'B' if self.ballOwner is 1 else 'b'
                else:
                    board += '-'
            board += '\n'
        print(board)
  def isInGoal_a(self, x, y):
        g1, g2 = self.goalPositions
        if (g1 <= y <= g2):
           
            if x == (self.w-1):
                self.done_a = bool(1)
                self.reward[0] = 20 # if the ball reaches the right goal post, then the rewards shall be 1
                return 0
        return -1
  def isInGoal_b(self, x, y):
        g1, g2 = self.goalPositions
        if (g1 <= y <= g2):

            if x == (self.w-1):
                self.done_b = bool(1)
                self.reward[0] = 20 # if the ball reaches the right goal post, then the rewards shall be 1
                return 0
        return -1

  def seed():
      return 0 
  def metadata(x):
      return 0 
  def legal_actions(self):
    return gym.spaces.Discrete(5)
  def close(self):
    pass