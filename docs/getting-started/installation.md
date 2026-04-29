# Installation

## Requirements

- Python 3.9 or later
- PyTorch 2.0+

## Option A — pip (recommended)

```bash
pip install epalea
```

With optional notebook dependencies:

```bash
pip install "epalea[notebooks]"
```

## Option B — conda

```bash
conda env create -f environment.yml
conda activate epalea
```

## Option C — from source

```bash
git clone https://github.com/epalea/epalea
cd epalea
pip install -e ".[notebooks,dev]"
```

## Verify installation

```bash
epalea info
```

## Jupyter kernel

```bash
python -m ipykernel install --user --name=epalea --display-name="epalea (LPF)"
jupyter notebook
```

## Colab

Add to the first cell of any notebook:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "epalea", "-q"])
```
