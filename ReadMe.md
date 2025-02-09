# Traveling Salesman Problem Research

Implementation of research related to different methods of solving the Traveling Salesman Problem. Performed for the Capstone Project as part of Temple University's Master's in Computer Science program.

## Setup

### With Python and Pip

* Install dependencies with `pip install -r requirements.txt`
* Install pyconcorde via the instructions on [GitHub](https://github.com/jvkersch/pyconcorde)

### With uv

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Install Python 3.13 via `uv python install 3.13.1`
* Install dependencies via `uv sync`
* Create a virtual environment with `uv venv --python 3.13.1`
* Activate the environment via `source .venv/bin/activate`
* Install dependencies via `uv sync`
* Install pyconcorde
  * Clone the repository on [GitHub](https://github.com/jvkersch/pyconcorde)
  * `cd` into the directory and run `uv pip install -e .`

## TSP Problem Data

Problems used are located in the `data` repo and can be called by name in the `main.py` script. Additional problems can 
be added by downloading a [tsplib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)-compatibly file and updating 
the `config.yaml`. Files can be decompressed with `gzip -d`.

## Run

Run the main script via

```shell
python main.py -p <problem name or comma-separated list of names>
```

By default, the problem(s) will run on all available algorithms. Specific algorithms can be specified with the `-a` parameter, either as a single algorithm abbreviation or a comma-separated list of algorithm abbreviations (see `main.py::ALGORITHMS` for full list).

Tests can be run via

```shell
pytest
```

## Workload stack

### Current Week

* Documentation and strong typing for existing functions
* Write midterm paper
* Get project in a submittable format for external testing

### Part II

* Algorithm tuning, mainly for Q-learning

## Future Work

* Dynamic Traveling-Salesman Problem
* Epsilon factor in ACO
