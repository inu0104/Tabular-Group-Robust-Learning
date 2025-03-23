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

Install required packages manually:

```bash
pip install torch pandas numpy scikit-learn
```

**Or** use the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```
torch
pandas
numpy
scikit-learn
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

| Model       | Paper                                                                                  |
|-------------|------------------------------------------------------------------------------------------|
| **NODE**     | [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) |
| **AutoInt**  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://dl.acm.org/doi/10.1145/3357384.3358028) |
| **DeepFM**   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/Proceedings/2017/0239.pdf) |
| **TabNet**   | [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) |

---

## ⚙️ Group-Robust Training Methods

| Method       | Description & Paper                                                                                              |
|--------------|------------------------------------------------------------------------------------------------------------------|
| **GroupDRO** | [Distributionally Robust Neural Networks for Group Shifts (Sagawa et al., ICLR 2020)](https://arxiv.org/abs/1911.08731) |
| **D3M**      | [Dynamic Data Debiasing via Matching (Park et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e0a3a51f0f7c1f55b9f404d9dfb148d3-Abstract-Conference.html) |

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
- Song et al. *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*. CIKM 2019.
- Guo et al. *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*. IJCAI 2017.
- Arik and Pfister. *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021.

### ⚙️ Methods
- Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts*. ICLR 2020. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- Park et al. *Dynamic Data Debiasing via Matching*. NeurIPS 2022.

---

💡 Feel free to open an issue or PR for questions or improvements!
