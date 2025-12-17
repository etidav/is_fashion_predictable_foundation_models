# is fashion predictable? The rise of foundation models
Code repository to reproduce results of the Medium article ["Is Fashion Predictable? The rise of foundation models"]()

## Code Organisation

The repository is organized as follows:

 - [foundation_model.py](src/is_fashion_predictable_foundation_models/model/foundation_model.py): File with all the foundation models tested in the artical.
 - [metrics.py](src/is_fashion_predictable_foundation_models/metrics.py): File with the three error metrics: MASE, SMAPE and MAPE.
 - [main.py](src/is_fashion_predictable_foundation_models/main.py): File reproducing the main experience of the media article.

## ⚙️ Installation

This project uses **uv** for dependency and environment management.

### 1. Clone the repository
```bash
git clone "link_of_the_repo"
cd is_fashion_predictable_foundation_models
```

### 2. Create a virtual environment (Python 3.11)
```bash
uv venv --python 3.11
```


### 3. Activate and install dependencies
```bash
source .venv/bin/activate
uv pip install -e '.[foundation]'
```

## 🚀 Reproducing the Results

```bash
cd src/is_fashion_predictable_foundation_models
python main.py
```
Control whether predictions are recomputed by editing `main.py`:

- `compute_prediction=True` → compute new predictions  
- `compute_prediction=False` → load precomputed predictions


