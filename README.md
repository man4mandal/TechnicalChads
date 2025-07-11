# TechnicalChads

Participants: Anirban Roy & Tarashankar Mandal
Repository: [https://github.com/man4mandal/TechnicalChads](https://github.com/man4mandal/TechnicalChads)

This project implements a gender classification pipeline with two modes:

* *Task A*: Predict gender (`Male` / `Female`) from test images.
* *Task B*: Face recognition and evaluation using class-map and ground-truth.

Execution is handled via a single script (`test.py`) with optional GPU acceleration.

---

## 📁 Repository Structure

```text
TechnicalChads/
├── test.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitattributes
├── models/
│   ├── TASK_A/
│   │   ├── best_model.pth
│   └── TASK_B/
│       ├── best_facenet_model.pth
│       ├── class_map.json
│       └── task_b_output.csv
└── test/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

---

## 🛠️ Setup Instructions

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 📆 Expected Input Format

The `test/` folder should contain images for inference:

```text
test/
├── img1.jpg
├── img2.jpg
└── ...
```

---

## ▶️ Command-Line Usage

### Task A: Gender Prediction

```bash
python test.py "models/TASK_A/best_model.pth" "test/"
```

### Task B: Model Evaluation

```bash
python test.py "models/TASK_B/best_facenet_model.pth" "test/" "models/TASK_B/task_b_output.csv" "models/TASK_B/class_map.json"
```

---

## 🔄 Reproducibility Notes

* The `test.py` script reproduces the same results as submitted.
* Uses provided model weights in the `models/` folder.
* No additional setup or manual intervention is required.
* Script supports both CPU and GPU (if available).

---

This README satisfies all Submission Execution Policy requirements.
