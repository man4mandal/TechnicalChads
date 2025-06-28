# Hackathon
made by Anirban.Roy
Sure! Here's your fully copy-pasteable `README.md`, complete from top to bottom, ready for GitHub:

````markdown
# 🧠 Hackathon AI Face Classifier

This project is a Python-based AI classification tool using deep learning for face and gender recognition. Built with `facenet-pytorch`, `torch`, and `customtkinter`, it features a minimal UI for batch image analysis.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
````

> Or download the ZIP and extract it to a desired location.

### 2. Open the Folder in Your Python IDE

Use VSCode, PyCharm, or any Python editor to open the extracted/cloned folder.

### 3. Open the Main Python File

Open:

```
Hackathon_Final Project.py
```

### 4. Set Up Python Virtual Environment (Python 3.8.10)

In the terminal:

```bash
python -m venv venv
```

Activate it:

* **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```
* **macOS/Linux:**

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

Update the following variables in `Hackathon_Final Project.py` to match your local file paths:

```python
face_model_path = r"your_local_path\face_model.pt"
gender_model_path = r"your_local_path\gender_model.pt"
train_folder = r"your_local_path\train_images"
```

Replace `your_local_path` with the actual path to the respective files and folders on your machine.

### 7. Run the Program

```bash
python "Hackathon_Final Project.py"
```

### 8. Choose Test Batch

When the interface opens:

* Select the folder named `test_batch`
* Click **OK**

### 9. ✅ View Final Results

The UI will display the classification output based on the batch of test images.

---

## 📁 Project Structure

```
📦 your-repo-name/
├── Hackathon_Final Project.py
├── face_model.pt
├── gender_model.pt
├── train_images/
├── test_batch/
├── README.md
```

---

## 🛠 Tech Stack

* Python 3.8.10
* PyTorch
* facenet-pytorch
* CustomTkinter
* Pillow

---

## 📸 Preview

1.
![App Preview](UI1.png)
2.
![App Preview](UI2.png)
3.
![App Preview](UI3.png)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributions

Feel free to fork the repo, submit issues, or make pull requests.

---

## 💡 Tip

Always use raw strings (`r"path\to\file"`) for Windows paths to avoid escape character issues.

---

Happy Building! ⚙️✨

```

Let me know your GitHub repo name and username if you'd like this personalized further.
```
