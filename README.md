# CS 8803-DRL class assignments for fall 2024
Homework/assignments for the fall 2024 class CS 8803-DRL at GaTech.

## Setup
Build and use the `cs8803drl` environment as follows (we recommend conda).

**NOTE:** Install the GPU version of PyTorch if preferred. If you already have the GPU version of PyTorch but would like the assignment code to use CPU, then set the `ONLY_CPU` flag to `True` under `src/utils.py`. Keep in mind that using the GPU might not necessarily be faster for the problems discussed.

### Python `venv`
Setup the python virtual environemnt (requires python `3.10`).
```bash
environment=cs8803drl
python -m venv "$environment"
source "$environment"/bin/activate
pip install -r requirements.txt
```

### Conda environment
Install conda on your system and then run
```bash
environment=cs8803drl
conda env create --name "$environment" --file=requirements.yaml
conda activate "$environment"
```
