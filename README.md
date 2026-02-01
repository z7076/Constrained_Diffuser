
## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

## Usage

Train a diffusion model with:
```
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```


Plan using the diffusion model with:
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```


## Acknowledgements

The diffusion model implementation is based on https://github.com/jannerm/diffuser and https://github.com/Weixy21/SafeDiffuser
