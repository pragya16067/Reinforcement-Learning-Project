# import all dependencies
from gym.envs.classic_control import MountainCarEnv

class MCEnv(MountainCarEnv):
    
    def __init__(self, partitions=6):
        super().__init__()
        self.name = "MountainCar"
        self.partitions = partitions
        self.n_states = partitions * partitions

        self.terminal_discrete_state = self.getDiscreteState([0.5, 0])

    # Dividing state space into 5 parts
    def split_pos_space(self, parts):
        x_range = [-1.2, 0.6]
        range_dict = {}
        l = x_range[0]
        delta = 1.8 / parts
        for i in range(parts):
            range_dict[i] = [l, l + delta]
            l = l + delta
        return range_dict

    def split_vel_space(self, parts):
        x_range = [-0.07, 0.07]
        range_dict = {}
        range_dict[0] = [-0.07, 0]
        range_dict[1] = [0, 0.07]
        return range_dict

    def get_state(self, range_dict_pos, range_dict_vel, x, v):
        disc_state = [0] * 2
        for state, sr in range_dict_pos.items():
            if x >= sr[0] and x < sr[1]:
                disc_state[0] = state
        for vel, vr in range_dict_vel.items():
            if v >= vr[0] and v < vr[1]:
                disc_state[1] = vel
        return disc_state


    def getDiscreteState(self, s):
        x, v = s
        parts = self.partitions
        vel_parts = 3
        range_dict_pos = self.split_pos_space(parts)
        range_dict_vel = self.split_vel_space(vel_parts)
        state = self.get_state(range_dict_pos, range_dict_vel, x, v)
        return (state[0]*vel_parts + state[1])


    def step(self, action):
        # Discretize state
        new_state, reward, self.done, info = super().step(action)
        state_1d = self.getDiscreteState(new_state)
        print(new_state)
        print(state_1d)

        if(state_1d==self.terminal_discrete_state):
            reward = 0
            self.done = True
        
        return state_1d, reward, self.done, info
        
    def reset(self):
        # Discretize state
        state = super().reset()
        state_1d = self.getDiscreteState(state)

        return state_1d
        