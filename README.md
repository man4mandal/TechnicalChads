# 🧠 Hackathon AI Face Classifier — TechnicalChads

Developed by **Anirban Roy** and **Tarashankar Mandal**, this project is a Python-based AI classification tool that uses deep learning for face and gender recognition. Built with `facenet-pytorch`, `torch`, and `customtkinter`, it features a minimal UI for batch image analysis.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/man4mandal/TechnicalChads.git
```

> Or download the ZIP and extract it to a desired location on your local machine.

### 2. Open the Folder in Your Python IDE

Use VSCode, PyCharm, or any preferred Python editor to open the extracted/cloned folder.

### 3. Open the Main Python File

Inside the folder, open:

```
Hackathon_Final Project.py
```

### 4. Set Up Python Virtual Environment (Python 3.8.10)

In the terminal:
```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 5. Install Required Dependencies

```bash
pip install Pillow
pip install customtkinter
pip install torch torchvision
pip install facenet-pytorch
```

### 6. Set File Paths in Code

Update the following variables in `Hackathon_Final Project.py` to match your local directory paths:

```python
face_model_path = r"your_local_path\face_model.pt"
gender_model_path = r"your_local_path\gender_model.pt"
train_folder = r"your_local_path\train_images"
```

> Replace `your_local_path` with the full path where the models and folders are stored on your system.

### 7. Run the Program

```bash
python "Hackathon_Final Project.py"
```

### 8. Choose Test Batch

When the UI launches:

- Select the folder named `test_batch`
- Press **OK**

### 9. ✅ View Final Results

The classification results for the test images will be displayed in the UI.

---

## 📁 Project Structure

```
📦 TechnicalChads/
├── Hackathon_Final Project.py
├── face_model.pt
├── gender_model.pt
├── train_images/
├── test_batch/
├── README.md
```

---

## 🛠 Tech Stack

- Python 3.8.10
- PyTorch
- facenet-pytorch
- CustomTkinter
- Pillow

---

## 📸 UI Preview

**1. Step1**
![App Preview 1](UI1.png)
**2. Step2**
![App Preview 2](UI2.png)
**3. Result**
![App Preview 3](UI3.png)

---

## 🎯 Sample Answer for Face Recognition and Gender Classification Training

Our AI system is trained using the **FACECOM** dataset, which contains over **5,000 face images** captured under challenging conditions like blur, low light, fog, and overexposure. We approached it as a two-task pipeline:

### Task A — Gender Classification:
We fine-tuned a **pre-trained ResNet-50** model by replacing the final layer to classify between **Male** and **Female**.

### Task B — Face Recognition:
We used a **FaceNet-based architecture** (`InceptionResnetV1`) to classify unique person identities. The final dense layer maps to all known individual IDs.

To ensure robustness, we applied **real-world augmentations** during training such as:

- Gaussian blur  
- Brightness and contrast variation  
- Motion blur  
- Shadow simulation  

Both models were trained using **PyTorch**, with:

- `CrossEntropyLoss` for classification  
- Optionally, `ArcFace` loss to enhance identity separation  
- **Adam optimizer**, cosine learning rate scheduler, and validation-based early stopping  

### Evaluation Metrics:
- **Gender:** Accuracy, F1-Score  
- **Identity:** Top-1 Accuracy, Macro-F1  

This ensures **fairness and high performance**, even under degraded visual conditions.

---

## 👨‍💻 Team TechnicalChads

| Member              | Role                          |
|---------------------|-------------------------------|
| Anirban Roy         | Programmer, AI & Backend       |
| Tarashankar Mandal  | UI Design, GitHub, Management  |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributions

Feel free to fork the repo, open issues, or make pull requests to improve this project.

---

## 💡 Tip

Always use raw strings (`r"path\to\file"`) in Python to avoid path formatting issues on Windows.

---

Happy Building! ⚙️✨
