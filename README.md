````markdown
# TechnicalChads

Participants: Anirban Roy & Tarashankar Mandal  
Repository: https://github.com/man4mandal/TechnicalChads

This project implements a gender classification pipeline with two modes:

- Task A: Predict gender (`Male` / `Female`) from test images.
- Task B: Face-Recognition

Execution is handled via a single script (`main.py`) with optional GPU acceleration.

---

## 📁 Repository Structure

```text
TechnicalChads/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitattributes
├── models/
│   ├── TASK_A/
│   │   ├── best_model.pth
│   │   └── task_a_labels.csv
│   └── TASK_B/
│       ├── best_facenet_model.pth
│       ├── class_map.json
│       └── task_b_output.csv
└── test/
````

---

## ⚙️ Setup & Execution

```bash
# Clone repository
git clone https://github.com/man4mandal/TechnicalChads.git
cd TechnicalChads

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Task A: Gender Prediction
python test.py "<model_path>" "<test_image_folder_path>"

# Task B: Model Evaluation
python test.py "<model_path>" "<test_image_folder_path>" "<task_b_output.csv_path>" "<class_map.json_path>"
```

---

## 📦 Dependencies

```text
torch
torchvision
pillow
scikit-learn
facenet-pytorch
```

---

## ✅ Notes

```text
- CLI-only execution; no notebooks.
- Auto-detects GPU if available.
- Maintain folder/file structure as-is.
```

```
```
