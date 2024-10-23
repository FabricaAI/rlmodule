# rlmodule
Flexible reinforcement learning models instantiators library


Now it only supports skrl, but is intended to be library agnostic - in later expansion

try other algos
shared separate model


<!--
# todo
# some envs have by default range on which they work, like pendulum -2, 2, need to propaget that in.
# cnn
# WRITE README tutorial
# Run & fix pre-commit
# annotate cfgs in modules - why doesn't work TYPE_CHECKING
# extensive comments
# Launch new version to pip
# Import new version in Isaac-lab
# lazy linear? what is it ?
# random model run add function back
-->


## How to run

### Install rlmodule from local code

- Make sure you are in base rlmodule dict.

- Start virtual env.
```
python3 -m venv venv
source venv/bin/activate
```
- Install library from local code
```
pip install -e .
```
Note: sometimes installation may fail, if there is a run/ dir present, you may need to remove it (TODO: fix)
```
rm -rf runs
```
### Run chosen example
```
python3 rlmodule/skrl/examples/gymnasium/skrl_gymnasium_pendulum.py
```


## Update new version to PIP

```
pip install build twine
```

```
rm -rf runs
python -m build
```

```
twine upload dist/*
```
