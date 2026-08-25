# 	SMRNET: Sector-aware Market Response Network for Robust Cross-Sectional Stock Return Ranking Prediction


This repository contains the official PyTorch implementation of **SMRNET**.

---

## 🌟 Model Architecture

SMRNET consists of four sequential stages:

### (1) Temporal Orthogonal Embedding
Historical time-series features of each equity are encoded via a multi-kernel CNN (`StockCNNEmbedding`) with kernel sizes of {1, 5, 11} to capture heterogeneous temporal patterns. A temporal self-attention mechanism (`TemporalSelfAttention`) then distills the most informative signals across the lookback window. The resulting representations are projected through an orthogonal linear transformation parametrized by the Cayley transform (`orthogonal_cayley`), preserving the geometric integrity of the representation space.

### (2) Sector-aware Decomposition
Inspired by the α/β decomposition of CAPM, each equity's latent representation is geometrically decoupled into a sector-common component (**BETA**) and an equity-specific residual component (**ALPHA**) via the `Neutralizer` module. The sector prototype is derived by average pooling over same-sector equities (`find_sector`). The two isolated components are then independently aggregated through dual GAT streams within `MarketGAT`:
- **Alpha-stream GAT**: aggregates idiosyncratic (ALPHA) signals among same-sector stocks
- **Beta-stream GAT**: aggregates systematic (BETA) signals among same-sector stocks

### (3) Signed Interaction Fusion (SIF)
The outputs of the dual GAT streams are fused via `SignedInteractionFusion`, which explicitly preserves both positive (+) and negative (−) directional cues. Cross-component interactions are computed separately for each polarity, selectively amplifying robust predictive signals while suppressing noise.

### (4) Prediction
The fused representations are mapped to predicted returns via `PositiveLinear`, a linear layer constrained by non-negative weights (via squared parametrization). This ensures that the directional signals disentangled during SIF are preserved without sign inversion, allowing each component's directional contribution to be directly reflected in the final output.

---

## 📂 Project Structure
| File/Folder | Description |
| :--- | :--- |
| `dataset/` | Data directory |
| `lib/Model.py` | Model architecture (`SMRNET`, `UnifiedModel`) |
| `lib/modules.py` | Core building blocks |
| `lib/datasetLoader.py` | Data loading utilities |
| `lib/Metric.py` | Evaluation metrics (IC, RIC, AR, SR) |
| `DataManager.py` | Dataset download & preprocessing via WRDS |
| `train.py` | Training logic and validation loops |
| `run.py` | Main entry point for training |
| `test.py` | Script for inference and evaluation |
| `scripts.sh` | Shell script for training pipeline |
| `scripts_test.sh` | Shell script for test pipeline |

---

## ⚙️ Environment Setup
The code is tested with **Python 3.11+**.

### 1. Requirements

```bash
pip install -r requirements.txt
```

Main Dependencies:
```
numpy==2.4.2
pandas==2.2.3
matplotlib==3.10.8
scikit-learn==1.8.0
scipy==1.16.3
tqdm==4.67.3
pyarrow==23.0.1
openpyxl==3.1.5
PyYAML==6.0.3
networkx==3.6.1
wrds>=3.1.0
```

> **PyTorch**: Install manually depending on your CUDA version.
> ```bash
> pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
> ```

> **TA-Lib**: Install the C library first before pip install.
> ```bash
> pip install ta-lib==0.6.4
> ```
> See: https://ta-lib.org/install/#linux-build-from-source

---

## 🗄️ Data Preparation

> ⚠️ **Note on Data Access:**
> This project uses **CRSP data** accessed via **WRDS (Wharton Research Data Services)**.
> A valid WRDS account with CRSP subscription is required.
> Data **cannot be publicly redistributed** due to licensing restrictions.

Once WRDS access is configured, download the stock universe as follows:

```bash
# Download Top N stocks by market cap (dynamic universe)
python DataManager.py --topk 500
```

This will generate the required dataset files under `./dataset/`.

---

## 🚀 Execution Pipeline

### 1) Training & Validation

```bash
# Method 1: Using shell script
bash scripts.sh

# Method 2: Using python command directly
python run.py \
    --model_type SMRNET \
    --random_seed 2026 \
    --learning_rate 0.0001 \
    --hidden_dim 64 \
    --dropout 0.3 \
    --train_epochs 100 \
    --patience 10 \
    --loss IC \
    --test_date 2025-12-31
```

### 2) Testing & Performance Evaluation

```bash
# Method 1: Using shell script
bash scripts_test.sh

# Method 2: Using python command directly
python test.py \
    --model_type SMRNET \
    --random_seed 2026 \
    --test_date 2025-12-31 \
    --setting seed2026_lr0.0001_hd64_gh1_nh1_dr0.3
```

### 4) Expected Output
`test.py` reports the final metrics in JSON format. The results are displayed in the console and automatically saved as `best_test_metric.json` in the model checkpoint directory.

```json
{
    "Z-MSE": 1.000014305114746,
    "IC": 0.032432116376288636,
    "IC_IR": 0.2516027716493248,
    "RIC": 0.030428126173850285,
    "RIC_IR": 0.2282068949986435,
    "AR_Top_10": 0.254485547542572,
    "SR_Top_10": 0.9122815552001586,
    "AR_Bot_10": 0.03715526685118675,
    "AR_Top_30": 0.24722115695476532,
    "SR_Top_30": 1.011315522214596,
    "AR_Bot_30": 0.07229121774435043,
    "AR_Top_50": 0.21683017909526825,
    "SR_Top_50": 0.9367116417787458,
    "AR_Bot_50": 0.0934288278222084
}
```


## License
For review purposes only. Full license information will be provided upon publication.
