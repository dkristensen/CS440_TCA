# UPS Capstone Project: Traffic Control Agent throuh Deep Reinforcement Learning
#### Author : [Drew Kristensen](https://github.com/dkristensen)
## Project Goal
The goal of this project is to create a policy learner to optimize the traffic flow in a non-virtual city environment. However, in order to accomplish this task, we will be using model based reinforcement learning approaches to tackle the end goal. By first training the architecture on an easily testable and repeatable enviornment, we can develop a model that will complete this task quicker and easier than one reliant on real world training data. One key feature in this project is keeping the inputs simple enough that they would be readily available in real life applications - that is to say the features used as inputs should be easy to observe. This project serves as the focus for the [CS440 "Capstone in Computer Science"](https://www.pugetsound.edu/academics/departments-and-programs/undergraduate/math-and-computer-science/course-descriptions-cs/) class at the University of Puget Sound.

## Packages Used
 * [TensorFlow](https://www.tensorflow.org/)
 * [SUMO](http://sumo.dlr.de/wiki/Simulation_of_Urban_MObility_-_Wiki)

## Process
This project will be completed using reinforcemenet learning techinques to develop an efficient policy to minimze time spent waiting at lights for cars in a city environment.  
Since this aproach has had success in other applications such as the board game GO with AlphaZero, teaching creatures mobility and actions (like in OpenAI's Transfer Learning or Deepmind's Locomotion experiments), or playing ATARI games (see OpenAI's DQN). The hope is that this project can be an example of yet another area that RL can be effective in, while creating a system that is both socially as well as environmentally beneficial. 

### Timeline
A predicted timeline for the project can be found in the Capstone Proposal document found at [capstone_proposal.pdf](capstone_proposal.pdf).


### Progress

##### Week 1
- [x] Get a working installation of the SUMO program, complete with full functioning SUMOPy tools.
- [ ] Put together a DDQN network to solve the Cartpole enviroment from the OpenAI gym.
- [x] Decide and write out the [state space and action space](state_action_spaces.pdf).
##### Week 2
- [ ] Put together a DDQN network to solve the Cartpole enviroment from the OpenAI gym
- [ ] Develop a program which changes the light signal at a single four way intersection that lets the directions with the most cars at the light through
- [ ] Brainstorm and write down the reward function