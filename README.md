Run test/enviornment_test.py to test the environment

Run test/train_test.py to test training. (This results in error)


Observation for single agent:
- Own position
- Own navigation goal
- Own task goal
- Other agents' positions, nav goals, and task goals (sorted by distance to self)

Action for single step:
[mode, (x,y,theta)]
mode = 0 -> follow path
mode = 1 -> plan path to (x,y,theta)