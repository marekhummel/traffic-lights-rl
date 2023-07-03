# traffic-lights-rl
Experimental RL project, to get into the topic, created as part of a opencampus.sh RL course. Based on stable-baselines3.

## Find optimal traffic light control with RL
-> When to switch which traffic light for most efficient traffic flow

### Simplified setting
- Two lights, only either of them can be green at any time
- New cars continuously arriving
- During green light, cars are passing 
- Reward for passing cars, penalty for long waiting cars
