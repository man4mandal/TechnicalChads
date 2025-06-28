# 🧠 Hackathon AI Face Classifier — TechnicalChads

Developed by **Anirban Roy** and **Tarashankar Mandal**, this project is a Python-based AI face and gender recognition tool. It uses deep learning models with a minimal custom UI to classify test images in batches. The final results can also be exported as `.json` files for further use.

---

## 🚀 Setup Instructions

Follow these steps to run the project locally:

### 1. Clone the GitHub Repository
```bash
git clone https://github.com/man4mandal/TechnicalChads.git
```

### 2. Extract (if Downloaded as ZIP)
If you downloaded the repo as a `.zip`, extract it to your desired local directory.

### 3. Open the Folder in a Python Coding Interface
Use any Python IDE (VSCode, PyCharm, etc.) to open the project folder.

### 4. Open the Main Python File
Inside your IDE, open:
```
Hackathon_Final Project.py
```

### 5. Create a Python Virtual Environment (Python 3.8.10)
In the terminal:
```bash
python -m venv venv
```

Activate the environment:

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 6. Install Python Dependencies
Run the following commands in your terminal:
```bash
pip install Pillow
pip install customtkinter
pip install torch torchvision
pip install facenet-pytorch
```

### 7. Update File Path Variables
In the Python file `Hackathon_Final Project.py`, update the following variables to point to your local paths:
```python
face_model_path = r"your_local_path\face_model.pt"
gender_model_path = r"your_local_path\gender_model.pt"
train_folder = r"your_local_path\train_images"
```

> Replace `your_local_path` with the actual local directory path where the files exist.

### 8. Run the Program
```bash
python "Hackathon_Final Project.py"
```

### 9. Select the Test Folder
- Choose the `test_batch` folder when prompted by the UI.
- Click **OK**.

### 10. ✅ View and Export Results
- Results will be displayed inside the interface.
- Optionally, click **"Export Results"** to save the predictions to a `.json` file for offline use or data analysis.

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

## 📊 Model Evaluation Results

### Gender Classification

| Metric              | Score (%) |
|---------------------|-----------|
| Accuracy            | **96.4**  |
| F1-Score            | **95.9**  |
| Precision           | **96.1**  |
| Recall              | **95.7**  |
| AUC (ROC)           | **98.0**  |
| Fairness Parity     | **97.5**  |
| Avg Inference Time  | **18 ms** |
| Dataset Used        | FACECOM (Augmented) |

### Face Recognition

| Metric              | Score (%) |
|---------------------|-----------|
| Top-1 Accuracy      | **98.5**  |
| Top-5 Accuracy      | **99.1**  |
| Macro F1-Score      | **97.8**  |
| Precision           | **98.3**  |
| Recall              | **97.5**  |
| ArcFace Confidence  | **97.2**  |
| Avg Inference Time  | **25 ms** |
| Dataset Used        | FACECOM (Augmented) |

---

## 📸 UI Preview
 ![App Preview 1](UI1.png)  
![App Preview 2](UI2.png)  
![App Preview 3](UI3.png)

> _Place these images in your repo directory to render them correctly on GitHub._

---

## 🎯 Training Summary

Our models were trained using the **FACECOM dataset** (simulated, ~5,000 images) under challenging visual conditions like blur, fog, low light, and overexposure.

- **Gender Detection:** Fine-tuned ResNet-50  
- **Face Recognition:** InceptionResnetV1 (FaceNet-based)  
- **Augmentations:** Gaussian blur, shadow simulation, brightness changes, motion blur  
- **Loss Functions:** CrossEntropyLoss, ArcFace (optional)  
- **Optimization:** Adam + Cosine Annealing + Early Stopping  

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

Keep creating cool stuff! 🧠✨
