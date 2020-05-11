



## UDACITY DEEP REINFORCEMENT LEARNING NANODEGREE 
### CAPSTONE 1 : NAVIGATION
#### SWASTIK NATH.
[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
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


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. Head over to the [Navigation.ipynb](https://github.com/swastiknath/rl_ud_1/blob/master/Navigation.ipynb) to take a look at the DQN Implementation and Training Procedure. 
