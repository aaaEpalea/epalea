# Notebook Environment Setup

## Option A — pip + venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install "epalea[notebooks]"
python -m ipykernel install --user --name=epalea --display-name="epalea (LPF)"
jupyter notebook
```

## Option B — conda

```bash
conda env create -f environment.yml
conda activate epalea
python -m ipykernel install --user --name=epalea --display-name="epalea (LPF)"
jupyter notebook
```

## Option C — Google Colab

Add to **Cell 0** of any notebook:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "epalea", "-q"])
import epalea
print(f"epalea {epalea.__version__} ready")
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_quickstart.ipynb` | Load `compliance-v1`, infer with `--mode both`, understand uncertainty |
| `02_training.ipynb` | All four pipeline stages + `train-full` shortcut |
| `03_inference.ipynb` | All three modes, metrics table, calibration curves |
| `04_custom_domain.ipynb` | Define a custom domain and run `train-full` |
