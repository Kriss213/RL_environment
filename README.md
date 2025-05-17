Run test/enviornment_test.py to test the environment

Run test/train_test.py to test training.


Observation for single agent:
- Own position
- Own navigation goal
- Own task goal
- Other agents' positions, nav goals, and task goals (sorted by distance to self)

Discrete action for single step:
0 - stay idle
1 - follow path

### Note:

```minimal_planner.launch.py``` must be started before training for path planning!

https://github.com/Kriss213/vikings_bot