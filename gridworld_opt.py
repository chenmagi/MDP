"""5x5 gridword"""

from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Function to add text to each grid cell
def add_text_to_grid(ax, text, x, y, font_size=12, font_weight='normal', text_color='black'):
    ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', fontsize=font_size, fontweight=font_weight, color=text_color)

def create_grid(shape=(5,5)):
    rows,cols = shape
    fig, ax = plt.subplots()
    ax.set_xticks(range(cols+1))
    ax.set_yticks(range(rows+1))
    ax.grid(True)

    # Set axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Draw lines for the grid
    for i in range(1, rows+1):
        ax.axhline(i, color='black', lw=2)
        ax.axvline(i, color='black', lw=2)

    return fig, ax


class TabularPolicy(defaultdict):
    def __init__(self,shape=(5,5)):
        super().__init__(lambda: [(0,1),(0,-1),(1,0),(-1,0)])
        self.rows,self.cols = shape
        return
    
    def select_action(self,state) -> tuple:
        num = len(self[state])
        idx = np.random.randint(num)
        return self[state][idx]

    def default_actions(self):
        return [(0,1),(0,-1),(1,0),(-1,0)]
    
    def current_actions(self,state):
        return self[state] # return list of available actions
    
    def get_uniform_prob(self,state): #equal probability for each action
        return 1.0/len(self[state])
    
    def update(self,state,actions) -> None:
        if not isinstance(actions,list):
            raise TypeError("need pack actions by list")
        self[state]=actions
        return 
    
    def next_state(self,state,move) -> tuple:
        if not isinstance(state,tuple):
            raise TypeError("need pack state by tuple")
        return tuple(x+y for x,y in zip(state,move))
    
    def get_repr(self,state):
        str=''
        for k in self[state]:
            if k == (0,1): str+='R'
            elif k==(0,-1): str+='L'
            elif k==(-1,0): str+='U'
            else: str+='D'
        return str
    
    def display(self):
        fmt_string="{:>4s} "* self.cols
        for r in range(self.rows):
            raw=[self.get_repr((r,c)) for c in range(self.cols)]
            print(fmt_string.format(*raw))    
        return
    
    
    
             

class TabularStateValue(defaultdict):
    def __init__(self,shape=(5,5)):
        super().__init__(lambda:0)
        self.policy = TabularPolicy(shape)
        
        self.rows,self.cols=shape
        self.stateA=(0,1)
        self.stateA_Prime=(4,1)
        self.stateB=(0,3)
        self.stateB_Prime=(2,3)
        
    
    def __evaluation(self,state):
        if not isinstance(state,tuple):
            raise TypeError("need pack state by tuple")
        state_value_sum=0
        for action in self.policy.current_actions(state):
            next_state=self.policy.next_state(state,action)
            reward=0
            y,x = next_state
            if x < 0 or y < 0 or x >= self.cols or y >= self.rows: # hit wall
                reward = -1
                next_state = state
            if state == self.stateA:
                reward = 10
                next_state = self.stateA_Prime
            elif state == self.stateB:
                reward = 5
                next_state = self.stateB_Prime
            #print('{}->{}, reward={}'.format(state,next_state,reward))
            state_value_sum+=self.policy.get_uniform_prob(state)*(reward+0.9*self[next_state])
        
        self[state]=state_value_sum
            
    def __update(self,state):
        action_statevalue_pairs = {}
        for action in self.policy.default_actions(): #allow to search whole set of actions
            ns = tuple(_x+_y for _x, _y in zip(state,action))
            y,x = ns
            if x < 0 or y < 0 or x >= self.cols or y >= self.rows:
                ns= state
            if state == self.stateA:
                ns = self.stateA_Prime
            elif state == self.stateB:
                ns = self.stateB_Prime    
            action_statevalue_pairs[action]=self[ns]
        #print('state={}, neighbor values={}'.format(state,action_statevalue_pairs.items()))
        max_value = max(action_statevalue_pairs.values())
        max_keys = [ key for key,value in action_statevalue_pairs.items() if int(value*100)==int(max_value*100)]
        
        #print('update state{} to policy{}'.format(state,max_keys))
        self.policy.update(state,max_keys)
                  
                                 
    def policy_evaluation(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.__evaluation((r,c))
                
                
    def policy_update(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.__update((r,c))
        
    def display(self, mode='console'):
        if mode == 'console':
            self.console_display()
        else:
            self.ui_display()
        
    
    def console_display(self):
        #print('display result')
        fmt_string="{:>5.1f} "*self.cols
        for r in range(self.rows):
            raw=[self[(r,c)] for c in range(self.cols)]
            print(fmt_string.format(*raw))
            
            
    def ui_display(self):
        fig, ax = create_grid()
        for row in range(self.rows):
            for col in range(self.cols):
                add_text_to_grid(ax, '{:>5.1f}'.format(self[(row,col)]), col, self.rows-row-1)
        
        plt.show()
        
        
            
    def __repr__(self):
        return f"TabularStateValue({self.default_factory},{dict(self)})"
    
            
            
            


    
def main():
    print('Policy Evaluation and Update demo')
    tabular=TabularStateValue()
    for t in range(20):
        tabular.policy_evaluation()
        tabular.policy_update()
        print('[{:>4s}]{}'.format(str(t+1),'-'*30))
        tabular.display()
        tabular.policy.display()
    tabular.display(mode='ui')    
    
        


if __name__ == '__main__':
    main()
    