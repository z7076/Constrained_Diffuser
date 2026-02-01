

# Installation
```
conda env create -f environment.yml
conda activate decdiff
export PYTHONPATH=/path/to/decision-diffuser/
```

# Training and Evaluation
To start training decision diffuser, run the following commmands

```bash
cd analysis
python train.py
```

To evaluate the trained model, run the following commmands
```bash
cd analysis
python eval.py
```

You can modify the training and evaluation configuration by appropriately changing `analysis/default_inv.py`.

# Acknowledgements

The codebase is derived from [Conditional diffuser repo](https://anuragajay.github.io/decision-diffuser/).