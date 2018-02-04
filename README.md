# UPS Capstone Project: Traffic Control Agent throuh Deep Reinforcement Learning
#### Author : [Drew Kristensen](https://github.com/dkristensen)
## Project Goal
The goal of this project is to create a policy learner to optimize the traffic flow in a non-virtual city environment. However, in order to accomplish this task, we will be using model based reinforcement learning approaches to tackle the end goal. By first training the architecture on an easily testable and repeatable enviornment, we can develop a model that will complete this task quicker and easier than one reliant on real world training data. One key feature in this project is keeping the inputs simple enough that they would be readily available in real life applications - that is to say the features used as inputs should be easy to observe. This project serves as the focus for the [CS440 "Capstone in Computer Science"](https://www.pugetsound.edu/academics/departments-and-programs/undergraduate/math-and-computer-science/course-descriptions-cs/) class at the University of Puget Sound.

One of the end goals for this project is for any city to be able to download their infrastructure (since SUMO and Open Street Maps work nicely together) and be able to learn the best agent for their city's traffic lights.

## Packages Used
 * [TensorFlow](https://www.tensorflow.org/)
 * [SUMO](http://sumo.dlr.de/wiki/Simulation_of_Urban_MObility_-_Wiki)

## Process
This project will be completed using reinforcemenet learning techinques to develop an efficient policy to minimze time spent waiting at lights for cars in a city environment.
Since this aproach has had success in other applications such as the board game GO with AlphaZero, teaching creatures mobility and actions (like in OpenAI's Transfer Learning or Deepmind's Locomotion experiments), or playing ATARI games (see OpenAI's DQN). The hope is that this project can be an example of yet another area that RL can be effective in, while creating a system that is both socially as well as environmentally beneficial. 

A predicted timeline for the project can be found in the Capstone Proposal document found [here](capstone_proposal.pdf).


### Progress

##### Week 1
- [x] Get a working installation of the SUMO program, complete with full functioning SUMOPy tools.
- [ ] Put together a DDQN network to solve the [Cartpole enviroment from the OpenAI gym](https://gym.openai.com/envs/CartPole-v0/).
- [x] Decide and write out the [state space and action space](state_action_spaces.pdf).
##### Week 2
- [x] Put together a DDQN network to solve the [Cartpole enviroment from the OpenAI gym](https://gym.openai.com/envs/CartPole-v0/).
- [x] Develop a program which changes the light signal at a single four way intersection that lets the directions with the most cars at the light through.
- [x] Brainstorm and write down the [reward function](reward.pdf).
##### Week 3
- [ ] Debug the implemented DDQN for the [cartpole problem](https://gym.openai.com/envs/CartPole-v0/).
- [x] Write a [program](Test_Scenario/state_getter.py) that collects the waiting time for all the cars in the system and returns the squared sum of waiting time for all vehicles in each lane.
- [x] Create a [python script](network_maker.py) to generate and populate with routes, a SUMO network in a rectangle with arbitrary number of intersections, arbitrary side lengths, and arbitrary lane numbers.
##### Week 4
- [ ] Set up the framework to create a DQN for a ***l*** way, ***n*** street intersection.
- [ ] Modify the above framework to use Double Q-learning