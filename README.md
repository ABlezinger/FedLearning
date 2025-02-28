# Federated Learning
This project aims to benchmark existing Federated Learning strategies.  

## Installation
### (Recommended) Setup new clean environment
Use a conda package manager.

## Installation
### (Recommended) Setup new clean environment
Use a conda package manager.

#### Conda
Subsequently run these commands, following the prompted runtime instructions:
```bash
conda create -n fed-learn python=3.10.16
conda activate fed-learn
pip install -r requirements.txt
```

## How to run
To run an experiment, specify the strategy. The code will run fedavg if none is specified.
Example:
```py
python Main.py --strategy fedyogi

```

## Used Hardware
| Compute/Evaluation Infrastructure    |                                      |
|:-------------------------------------|--------------------------------------|
| Device                               | MacBook Pro M3 Pro 14-Inch                  |
| CPU                                  | M3 Pro |
| GPU                                  | -                                    |
| TPU                                  | -                                    |
| RAM                                  | 18 GB RAM                       |
| OS                                   | Sonoma 14.5                        |
| Python Version                       | 3.10.16                      |


