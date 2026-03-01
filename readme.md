
---

```markdown
# Sleep Breathing Irregularity Detection

This project focuses on detecting abnormal breathing patterns during sleep using physiological signals such as nasal airflow, thoracic movement, and SpO₂.

The dataset consists of recordings from multiple participants, each containing several hours of sleep data along with annotated breathing events.

---

## Project Structure

```

.
├── scripts/
│   ├── vis.py
│   ├── create_dataset.py
│   ├── train_model.py
│
├── models/
│   └── cnn1.pkl
│
├── Visualizations/
├── Dataset/
├── Data/
│
├── metrics.yaml
├── requirements.txt
└── README.md

````

---

## Important Note

Due to size constraints, the following directories are excluded using `.gitignore`:

- `Data/`
- `Dataset/`

You will need to manually place the raw data inside the `Data/` folder before running the pipeline.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## Workflow

### 1. Visualization

To inspect raw signals and annotated events:

```bash
python scripts/vis.py -name Data/AP01
```

This generates plots and saves them in the `Visualizations/` directory.

---

### 2. Dataset Creation

To preprocess signals and generate the dataset:

```bash
python scripts/create_dataset.py -in_dir Data -out_dir Dataset
```

This step performs:

* Signal filtering (0.17–0.4 Hz band)
* Time alignment of signals
* Windowing (30 seconds, 50% overlap)
* Label assignment based on event overlap

Output:

* `Dataset/dataset.pkl`

---

### 3. Model Training and Evaluation

Train the model using Leave-One-Participant-Out cross-validation:

```bash
python scripts/train_model.py
```

This will:

* Train a 1D CNN model
* Evaluate on unseen participants
* Handle class imbalance using weighted loss

Outputs:

* `metrics.yaml` (performance + hyperparameters)
* `models/cnn1.pkl` (trained model)

---

## Method Overview

* Signals are filtered to retain breathing-related frequencies
* Data is segmented into fixed-length windows
* Labels are assigned based on overlap with annotated events
* A 1D CNN is used for classification
* Evaluation is performed using participant-wise cross-validation

---

## Outputs

* `metrics.yaml`
  Contains accuracy, precision, recall, and hyperparameters

* `models/cnn1.pkl`
  Contains trained model weights and metadata

* `Visualizations/`
  Contains signal plots with event annotations

---

## Notes

* Data is normalized before training
* Class imbalance is handled using weighted loss
* Model is implemented in PyTorch
* Evaluation avoids data leakage by separating participants

---

```

---


```
