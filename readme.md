## Dependencies:
1. Python3
2. Pipenv

## How to use:
1. Create a .venv dir. pipenv will use this dir for packages and scripts
```
mkdir .venv
```
2. Install requirements from package file (Pipfile)
```
pipenv install
```
3. Enter pipenv shell to run scripts
```
pipenv shell
```
4. Now use python to run scripts
```
python src/<script>.py
```

## Train and test the autoencoders:

- _train scripts are for training.  
The models will be saved at /models dir, as a .keras file.  
You only need to train models once

- _ae scripts are for testing.  
They will read the .keras files at /models dir
