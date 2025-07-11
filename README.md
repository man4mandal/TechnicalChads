````markdown
# TechnicalChads

**Participants:** Anirban Roy & Tarashankar Mandal  
**Repository:** https://github.com/man4mandal/TechnicalChads

This project implements a gender classification pipeline supporting two modes:

- **Task A**: Predict gender (`Male` / `Female`) from test images.
- **Task B**: Evaluate performance using a ground-truth CSV and class-map JSON.

Both tasks are executed via a single script (`main.py`), powered by PyTorch-based models with GPU support when available.

---

## 📁 Repository Structure

```text
TechnicalChads/
├── LICENSE
├── .gitattributes
├── README.md
├── main.py
├── requirements.txt
├── models/
│   ├── TASK_A/
│   │   ├── best_model.pth
│   │   └── task_a_labels.csv
│   └── TASK_B/
│       ├── best_facenet_model.pth
│       ├── class_map.json
│       └── task_b_output.csv
└── test/
```


---

## ⚙️ Setup & Execution Flow

```bash
# 1. Clone the repository
# ------------------------
git clone https://github.com/man4mandal/TechnicalChads.git
cd TechnicalChads

# 2. Create and activate virtual environment
# ------------------------------------------
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
# ------------------------
pip install -r requirements.txt

# 4. Run Task A – Gender Prediction
# ----------------------------------
python main.py "model_path" "test_image_folder_path"

# 5. Run Task B – Evaluation
# ---------------------------
python main.py "model_path" "test_image_folder_path" "task_b_output.csv_path" "class_map.json_path"
```

---

## 📦 Dependencies (`requirements.txt`)

```text
torch
torchvision
pillow
scikit-learn
facenet-pytorch
```

---

## ✅ Important Notes

```text
- Execution is CLI-only—no notebooks or manual configuration.
- Script auto-detects GPU if available.
- Results must match original submission exactly—any deviation may cause disqualification.
- Folder and file names are case-sensitive and must remain unchanged.
```
````
