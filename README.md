# rlmodule
Flexible reinforcement learning models instantiators library


## How to run

### Install rlmodule from local code

- Make sure you are in base rlmodule dit.

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



