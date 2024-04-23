# Introduction

This is a repository that showcases how we can perform random sweeps of hyper-parameters in Modal. Here are some instructions on how to run it.

## Instructions

1. Create a new virtual environment

```
uv venv venv && source venv/bin/activate
```

2. Install the required dependencies

```
uv pip install -r requirements.txt
```

3. Run the script

```
modal run main.py
```

This will start our script that will take in the combination of parameters you've defined inside `main.py` and generate random permutations of combinations. We then create new containers on the fly so that we can run these jobs in parallel. Results will then be logged into the `paramsearch` repository ( see the sample `.md` file and the `.csv` file for an example of what the output will look like).
