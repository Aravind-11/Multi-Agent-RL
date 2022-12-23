## Two Player Soccer

Welcome to Two Player Soccer, a zero-sum game where two agents compete against each other to score the most goals in a soccer setting.

### Game Rules

* At the start of the game, one of the agents is randomly given the ball.
* The goal of the agent with the ball is to move it to the opposing team's goal post and score a point.
* If the agent with the ball comes in contact with the other agent, the ball is transferred to the other agent. The agent that loses the ball receives a penalty.

### Running the code

To run the code, make sure you have the required dependencies installed. You can install them by running:

```
pip install -r requirements.txt

```
Then, run the main script using:

```
python main.py
```

### Dependencies

The code has the following dependencies:

* os
* glob
* time
* datetime
* torch
* gym
* roboschool
* pybullet_envs
* numpy

Make sure you have these packages installed before running the code. You can install them using pip.

I hope you enjoy playing Two Player Soccer! Let me know if you have any other questions.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc



# AI-Gaming

![](https://github.com/Aravind-11/Multi-Agent-RL/blob/main/Screenshot%202021-11-20%20at%209.29.16%20PM.png)
# Problem Statement
Two RL agents, player A and player b (capital A denoting the player has the ball ) compete in an 8*8 grid environment, having 5 discrete actions ( up, down, left, right, and stationary ) to score a goal at the coordinate defined in the environment.

The agents not only have to deal with maximizing their reward in the environment but also have to deal with another RL Agent (multi-agent problem in RL).

**State-space** :
We would like to add the following variables in the state space: Ball Owner, x coordinate of player, y coordinate of player, x coordinate of the opponent, y coordinate of the opponent, (distance of the player from goal post in the x-axis ), (distance of the player from the goal in the y axis ), score of our bot, score of opponent .

**Reason** :
We want our agent to know who is the ball owner, and the distances from the goal post throughout the entire training process so that it can move to the goal post faster. Also the problem of boundaries, that is ( if our agent reaches an edge and the action sampled from our neural network forces it to move out which is nullified by the environment, the action is wasted in this case. This problem is also addressed by the last two variables in the state space that calculates the distance of the agent from the goalposts ). x and y coordinates are given as usual for the training process for offense and defense maneuvers​.

**Action space**:
We are using the actions which are possible in the environment as the action space for the problem, namely 0 - Stationary, 1 - move up, 2 - move right, 3 - move down, 4 - move left

**Rewards** :
We are treating them as values that need to be tuned for training our bot. We will just explain the rewarding system in detail here.

**Reward** - 
For Goal - a numpy array of shape (1,2) , where the first column represents the reward for our bot and the second column represents reward for the opponent (since we are using self play) .

**Penalize** - 
For losing the ball to the opponent, if our opponent is the ball owner, our bot has done self-goal, our opponent scoring goal.
Reasons for above penalizing and rewards :

**Reward** - 
We expect our agent to maximize its expected return from the environment which is done by scoring more goals. We would also like to reward the agent for having the ball as a means of telling the agent to not give the ball to the opponent

**Penalize** - 
Passing the ball to the opponent is a move that our agent cannot afford to do, and self-goal is discouraged too so that our agent is not encouraged to move backward and treat that as a goal
# Model

### The Proximal Policy Optimisation (PPO) engine:
PPO, developed by OpenAI in 2017, is a state-of-the-art RL algorithm. It updates the current version of the ​network​ being trained and fires a ​callback​ that saves the network to the bank if the current version has outperformed previous versions.

### Method
Two PPO models each for agent_A and agent_B are trained independently with a joint state space, independent action space and rewards. The idea is adopted from this [paper](https://proceedings.neurips.cc//paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Paper.pdf). 

 
# References
#### Training

https://youtu.be/gqX8J38tESw?t=2999

#### PPO
https://arxiv.org/abs/1707.06347 

https://openai.com/blog/openai-baselines-ppo/ 

https://www.youtube.com/watch?t=14m1s&v=gqX8J38tESw&feature=youtu.be
      
