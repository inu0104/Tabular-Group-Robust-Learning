# 📊 Tabular Group-Robust Learning

This repository provides a unified framework for training **group-robust models on tabular datasets** using various deep learning architectures.  
The focus is on **binary classification** tasks with **multiple predefined groups**, supporting both standard and fairness-aware training methods.

---

## 🗂️ Project Structure

<details>
<summary>Click to expand</summary>

```
project/
├── dataset/                    # Raw and processed data
├── models/                     # Model architectures
│   ├── node.py
│   ├── autoint.py
│   ├── deepfm.py
│   ├── tabnet.py
│   └── __init__.py
├── utils/                      # Utilities for training and evaluation
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluation.py
│   ├── alignment.py
│   ├── compute_gradient.py
│   ├── compute_tau.py
│   └── __init__.py
├── method/                     # Fairness-aware training methods
│   ├── d3m.py
│   ├── group_dro.py
├── experiments/                # Example configuration files
│   ├── adult_autoint.json
├── main.py                     # Main training entry
├── requirements.txt
└── README.md
```

</details>

---

## 🛠️ Installation

Install with `pip`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
pandas
numpy
scikit-learn
```

✅ All packages are compatible with standard Python environments (tested on Python 3.8+).

---

## 📥 Dataset

To run the experiments, download the dataset zip file from the link below:

🔗 [Download Dataset (Google Drive)](https://drive.google.com/file/d/157ZB-alPtSQBzZNuBzbvy4bkbMcUL0tU/view?usp=drive_link)

Once downloaded, extract the contents and place the `dataset/` folder **in the root directory of the project**:

```
project/
├── dataset/                ✅ Place extracted folder here
├── models/
├── utils/
├── method/
├── main.py
└── ...
```

After setup, you can run experiments using:

```bash
python main.py --config experiments/adult_autoint.json
```

---

## 📑 Dataset Format

Each dataset should include:

- `target`: binary label (0 or 1)
- `group`: integer indicating group membership (e.g., race, gender)
- `sample_id`: unique identifier
- All other columns: numerical or one-hot encoded features (`float32`)

Datasets are preprocessed via `utils/data_loader.py` and wrapped using `GroupDataset`.

---

## 🧠 Supported Models

| Model       | Description & Paper                                                                                  |
|-------------|-------------------------------------------------------------------------------------------------------|
| **NODE**     | [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (Popov et al., 2019)](https://arxiv.org/abs/1909.06312) |
| **AutoInt**  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (Song et al., 2019)](https://arxiv.org/abs/1810.11921) |
| **DeepFM**   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (Guo et al., 2017)](https://www.ijcai.org/Proceedings/2017/0239.pdf) |
| **TabNet**   | [TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, 2019)](https://arxiv.org/abs/1908.07442) |

---

## ⚙️ Group-Robust Training Methods

| Method       | Description & Paper                                                                                              |
|--------------|------------------------------------------------------------------------------------------------------------------|
| **GroupDRO** | [Distributionally Robust Neural Networks for Group Shifts (Sagawa et al., ICLR 2020)](https://arxiv.org/abs/1911.08731) |
| **D3M**      | [Data Debiasing with Datamodels (D3M): Improving Subgroup Robustness via Data Selection (Kim et al., 2024)](https://arxiv.org/abs/2406.16846) |

---

## 🚀 Running Experiments

Run with:

```bash
python main.py --config experiments/adult_autoint.json
```

**Example config:**

```json
{
  "model_type": "autoint",
  "method": "group_dro",
  "dataset": "adult",
  "model_params": { "hidden_dims": [64, 32], "dropout": 0.1 },
  "train_params": { "lr": 0.001, "epochs": 10 }
}
```

---

## 📚 Citations

### 🧠 Models
- Popov et al. *Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data*. [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)
- Song et al. *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*. [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)
- Guo et al. *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*. IJCAI 2017. [PDF](https://www.ijcai.org/Proceedings/2017/0239.pdf)
- Arik and Pfister. *TabNet: Attentive Interpretable Tabular Learning*. [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)

### ⚙️ Methods
- Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- Kim et al. *Data Debiasing with Datamodels (D3M): Improving Subgroup Robustness via Data Selection*. [arXiv:2406.16846](https://arxiv.org/abs/2406.16846)

---

💡 Feel free to open an issue or PR for questions or improvements!
