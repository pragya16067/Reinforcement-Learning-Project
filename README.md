# Reinforcement Learning
A repo for the project of course Reinforcement Learning (COMPSCI-687). This includes the following algorithms implemented for toy domains from open AI gym.

### Project structure

#### Algorithms
```
Algorithms:
    - N-Step_SARSA.ipynb has the code for implementing episodic N-Step SARSA for Mountain Car and Acrobat domains. It uses epsilon-soft policy for exploration. Sutton's Tiling code has been used for encoding states.

    - REINFORCE.ipynb has the code for implementing REINFORCE algorithm with whitened rewards as baseline for Cartpole and Acrobat domains.

    - PrioritizedSweeping.py has the code for implementing Q-Learning with prioritized sweeping for Gridworld and Mountain car domains. 

    - utility_lib has the code for Tiling encoding taken from http://incompleteideas.net/tiles/tiles3.py-remove
```

#### Environment
```
Environments:
    - Gridworld: Implemented gridworld environment from scratch using Environment base class from Open AI Gym.
    
    - MountainCar: Extension of Gym's Mountain car domain which allows discretization of the state space for tabular policy algotithms.
```


