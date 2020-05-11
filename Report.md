## UDACITY DEEP REINFORCEMENT LEARNING NANODEGREE 
### CAPSTONE 1 : NAVIGATION
#### SWASTIK NATH.

#### A Few Words About the Environment:
The goal of this capstone was to train an inteligent model which will be able to navigate around a big square world and collect yellow bananas and avoid obstacles like blue bananas. 

#### Reward Design of the Environment:

The reward function acts like follows:

 `+1` : Collecting a Yellow Coloured Banana.

 `-1` : Collecting a Blue Coloured Banana. 

With the reward function above we can understand that the agent should collect as many as yellow bananas as possible while avoiding running into the blue ones. 

#### The State Space of the Environment:

The state space of the environment has 37 dimensions in total. It containes various information about the Agent such as it's velocity, along with Ray-based perception of objects around agent's forward direction. The environment is episodic. 

#### The Action Space of the Environment:

Given the state space of 37 dimensions the agent can choose one of the following actions:

 `0` : Move Forward

 `1` : Move Backward

 `2` : Turn Left

 `3` : Turn Right

As the environment suggests a average reward of +13 and above for over 100 episodes will enact as the solution of the problem. 

### Algorithmic Approach:

#### The Baseline:

In order to gain an intuition abou the underlying environment and it's reward structure we first define a solution where the **Uniform Random Policy** with equiprobable actions is defined. In this case, by running the policy for quite a few times we notice that with Random Actions the agent can achieve +2 to -2 range of rewards and due to randomness it is not reproducible. So, by following the Uniform Random Policy will not solve the environment. 

#### Resolving with Deep Q Network:

In order to get the model solved we implemented a **Deep Q Network** which uses two Feed forward neural networks. Formally, DQN is a multi-layered neural network that for a given state `s` outputs a vector of action values. The two important ingredients of DQN is the use of **Target Network** and the use of **Experience Replay Buffer**. The Target network is quite the same as online network except its parameters are copied in every pre-specified steps from the online network. To build up intuition we feed the neural network the **State** of the environment and the neural network actually acts as a mapping function which maps the **States** to corresponding best-suited action. Upon this action we then perform Epsilon Greedy Policy where in case of the random number being larger than the timestep decaying epsilon, we select the maximum of actions or on being lower we select a random action thereby balancing off the exploration-exploitation tradeoff. With incresing timesteps the explorations decrease and the epsilon gets fixed thereafter. In DQN we use a total of 2 Action value (Q) learning network first one the online action-value estimation network(local) and the second one is the target action-value estimation network to calculate the error between the target action value and the expected action value. We use Derivatives to propagate the approximation errors across the previous layers of the deep neural network. The Action-Value function with regard to the function apporoximator and the Q-Learning Target for DQN is as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\fn_jvn&space;Q_\pi(s,a)\equiv&space;E[R_1&plus;\gamma&space;R_2&plus;...|&space;S_0=s,A_0=a,\pi&space;]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\fn_jvn&space;Q_\pi(s,a)\equiv&space;E[R_1&plus;\gamma&space;R_2&plus;...|&space;S_0=s,A_0=a,\pi&space;]" title="Q_\pi(s,a)\equiv E[R_1+\gamma R_2+...| S_0=s,A_0=a,\pi ]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\fn_jvn&space;Y_t^{DQN}=R_{(t&plus;1)}&plus;&space;\gamma&space;max_aQ(S_{(t&plus;1)},a,\theta_t^{-}&space;)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\fn_jvn&space;Y_t^{DQN}=R_{(t&plus;1)}&plus;&space;\gamma&space;max_aQ(S_{(t&plus;1)},a,\theta_t^{-}&space;)" title="Y_t^{DQN}=R_{(t+1)}+ \gamma max_aQ(S_{(t+1)},a,\theta_t^{-} )" /></a>

![Deep Q Network Training](https://github.com/swastiknath/rl_ud_1/blob/master/images/dqn_normal.png)  ![From DQN Saved CheckPoint](https://github.com/swastiknath/rl_ud_1/blob/master/images/dqn_normal2.png)

We use the **Double Q Learning** Algorithm to mitigate potential overestimation of the Action(Q) values. In the case of Double Q Learning algorithm two action-value functions are learned by assigning each experience randomly to update one of the two value functions leading to two sets of weights. Among these sets of weights one is used to determine the greedy policy and the another one is used to determine the correspoding action-value. Q-Learning target can be specified as the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\fn_jvn&space;Y_t^Q=R_{(t&plus;1)}&plus;&space;\gamma&space;Q(S_{(t&plus;1)}&space;argmax_aQ(S_{(t&plus;1)},a,\theta&space;_t&space;);\theta&space;_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\fn_jvn&space;Y_t^Q=R_{(t&plus;1)}&plus;&space;\gamma&space;Q(S_{(t&plus;1)}&space;argmax_aQ(S_{(t&plus;1)},a,\theta&space;_t&space;);\theta&space;_t)" title="Y_t^Q=R_{(t+1)}+ \gamma Q(S_{(t+1)} argmax_aQ(S_{(t+1)},a,\theta _t );\theta _t)" /></a>

![](https://github.com/swastiknath/rl_ud_1/blob/master/images/double_dqn.png) ![](https://github.com/swastiknath/rl_ud_1/blob/master/images/double_dqn_2.png)

We have also implemented the **Dueling Network** Architecture for our algorithm here for better evaluation of the model-free policy. Dueling Networks represents two seperate estimators one for the state-value function and another one is the state-dependent action advantage function. The main benefit of these estimators is to generalize the learning across different actions. The Action Value Estimation can be written as the following: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\fn_jvn&space;Q(s,&space;a;&space;\theta&space;,&space;\alpha&space;,&space;\beta&space;)&space;=&space;V(s;&space;\theta&space;,&space;\beta&space;)&space;&plus;&space;A(s,&space;a;&space;\theta&space;,&space;\alpha&space;)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\fn_jvn&space;Q(s,&space;a;&space;\theta&space;,&space;\alpha&space;,&space;\beta&space;)&space;=&space;V(s;&space;\theta&space;,&space;\beta&space;)&space;&plus;&space;A(s,&space;a;&space;\theta&space;,&space;\alpha&space;)" title="Q(s, a; \theta , \alpha , \beta ) = V(s; \theta , \beta ) + A(s, a; \theta , \alpha )" /></a>

![](https://github.com/swastiknath/rl_ud_1/blob/master/images/duel_dqn.png)  ![](https://github.com/swastiknath/rl_ud_1/blob/master/images/duel_dqn2.png)

#### Performance Scenario:

Let's take a look at the efficiency of diiferent algorithms to solve the environment in terms of their checkpoint retrival, and how faster each can reach the converge to the optimal learning optimal policy.

| Name of the Algorithm | Episodes Taken to Solve | Reward at the End | Highest Reward(From CheckPoint) | Lowest Reward(From CheckPoint)|
|-----------------------|-------------------------|-------------------|----------------|------------|
|Deep-Q-Network   | 331  | 13.01 | 18.00 | 9.71 |
|Double Deep-Q_Network with Duel Architecture | 307 | 13.00 | 16.00 | 6.00|
|Double Deep-Q-Network | 293 | 13.01 | 19.00 | 12.75|
|Deep-Q-Network with Duel Architecture | 268 | 13.05 | 16.00 | 6.00|  

#### Selected Hyperparameters:
The research paper of the DQN [avilable here](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) suggested the following hyperparameters for the DQN implementations for the Atari Games. The Hyperparameters works quite well in this scenario too with a few little or no tweaks. 
| Name of the Hyperparameter | Value |
|----------------------------|--------------|
| Discount Factor(γ)            |    0.99      |
| Learning Rate for Adam Optimizer |  5e-4     |
| Initial Epsilon                      |    1.0       |
| Epsilon Decay Rate          |    0.98     |
| Fixed Terminal Epsilon      |   0.02       |
|Batch Size | 64    |
| No. of Hidden Layers in the 1st Layer of Q-Network | 64   |
| No. of Hidden Layers in the 2nd Layer of Q-Network |   64 |
| Experience Replay Buffer Size |   1e5
| Q-Network Parameters Update Frequency from Experience Replay Buffer | 4  |
| Tau (τ) For Soft-Update of the Target Network                         |   1e-3          |

#### A Note About the Future:

Although the model gets solved by the DQN with Dueling Architecture much faster however, we get lower average score in terms of playing with the saved trained weights. However the Double DQN converges a little bit slower than the Dueling DQN but is able to achieve higher scores in terms of playing with the saved trained weights. So, for this task scenario, a Double DQN architecture will be more performant than the other variants. 
