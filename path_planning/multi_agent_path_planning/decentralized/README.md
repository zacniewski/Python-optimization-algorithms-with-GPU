### Decentralized solutions

In this approach, it is the responsibility of each robot to find a feasible path. 
Each robot sees other robots as dynamic obstacles, and tries to compute a control velocity which would avoid collisions with these dynamic obstacles.  
Usage:  
```bash 
python3 decentralized.py -f <filename> -m <mode>
```

> Flags:  
`-m`: mode of the obstacle avoidance, options are `velocity_obstacle` or `nmpc`,  
`-f`: filename, in the case you want to save the animation
1. Velocity obstacles  

```bash
python3 decentralized.py -f velocity_obstacle/velocity_obstacle.avi -m velocity_obstacle
```

Article [The Hybrid Reciprocal Velocity Obstacle](http://gamma.cs.unc.edu/HRVO/HRVO-T-RO.pdf) was used. 

2. Nonlinear Model-Predictive Control  
```bash
python3 decentralized.py -m nmpc
```
Article [Nonlinear Model Predictive Control for Multi-Micro Aerial Vehicle Robust Collision Avoidance](https://arxiv.org/abs/1703.01164) was used.


