# FedLearning

# Advanced Deep Reinforcement Learning
This project aims to combine the exploration strategy offered by Random Network Distillation (RND) with the robustness of Proximal Policy Optimization (PPO). As previous attempts improved smaller state-space environments, this project is thought to solve the milestones from Crafter environment which is a 2D abstraction of Minecraft. The main challenge in this project is the utilization of intrinsic rewards while the base environment has incorporated milestones which are essential for the reward function and solving the environment. 

## Results
First plot is vanilla PPO on 500 episodes
![alt text](https://github.com/arzx/adrl-project/blob/main/src/plots/baselines_final/mean_scores_with_std_dev.png)
Second plot is PPO with RND
![alt text](https://github.com/arzx/adrl-project/blob/main/src/plots/ppo_rnd_final/mean_scores_with_std_dev.png)
## Installation
### (Recommended) Setup new clean environment
Use a conda package manager.

#### Conda
Subsequently run these commands, following the prompted runtime instructions:
```bash
conda create -n crafter python=3.12.15
conda activate crafter
pip install -r requirements.txt
```

## How to run
`export PYTHONPATH=$(pwd)/src:$PYTHONPATH`
#### baselines ppo
`python src/agents/baselines.py --steps 500`


#### To run without seeds and log the policies for comparison
`src/agents/ppo_rnd/with_policies.py --steps 500`


#### ppo_rnd version with seeds
`python src/agents/ppo_rnd/new_ppornd.py --steps 500`

#### plotting 
The plot `scores_baselines.png` show each seeds trajectory to inspect robustness and correct episode length. The plot `mean_scores_with_std_dev.png` shows the mean achievement over 500 episodes with the respective confidence interval across all seeds.

Change path of to the json of your scores.json file from logdir, resulting plots are found in src/plots directory
python src/utils/plotting.py


## Rewards in Crafter
Reward: The sparse reward is +1 for unlocking an achievement during the episode and -0.1 or +0.1 for lost or regenerated health points. Results should be reported not as reward but as success rates and score. Episodes end automatically after 10.000 steps if the agent survives that long.

Success rates: The success rates of the 22 achievemnts are computed as the percentage across all training episodes in which the achievement was unlocked, allowing insights into the ability spectrum of an agent.

## Used Hardware
| Compute/Evaluation Infrastructure    |                                      |
|:-------------------------------------|--------------------------------------|
| Device                               | MacBook Pro M3 Pro 14-Inch                  |
| CPU                                  | M3 Pro |
| GPU                                  | -                                    |
| TPU                                  | -                                    |
| RAM                                  | 18 GB RAM                       |
| OS                                   | Sonoma 14.5                        |
| Python Version                       | 3.12.15                      |

## Further Work
To reduce the necessary computational resources to reach 1 million episodes, it is recommended to transfer this approach to Craftax [3]. It is an implementation of Crafter in Jax and offers a faster performance. Furthermore each achievement can be inspected separately.

## Sources
[1] Nugroho, W. (2021). Reinforcement Learning PPO RND [Source code]. GitHub. https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd \
[2] Hafner, D. (2021). Benchmarking the Spectrum of Agent Capabilities. arXiv preprint:2109.06780 \
[3] Matthews, M. et al. (2024). Craftax: A Lightning-Fast Benchmark for Open-End Reinforcement Learning. International Conference on Machine Learning (ICML).
[4] Bradbury, J. et al (2018). JAX: composable transformations of {P}ython+{N}um{P}y programs. http://github.com/google/jax. 
[5] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
