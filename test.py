import os
import csv
import json
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Female', 'Male']

def load_class_map(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['filename']] = row['class']
    return gt

def load_model(model_path, num_classes, backbone="resnet34"):
    model = models.resnet18(pretrained=False) if backbone == "resnet18" else models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()
    return pred

def evaluate_with_gt(model_path, test_folder, label_csv_path, class_map_path):
    class_map = load_class_map(class_map_path)
    idx_to_class = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    model = load_model(model_path, num_classes, backbone="resnet18")
    ground_truth = load_ground_truth(label_csv_path)

    y_true = []
    y_pred = []

    for filename in sorted(os.listdir(test_folder)):
        if filename.lower().endswith('.jpg') and filename in ground_truth:
            img_path = os.path.join(test_folder, filename)
            true_class = ground_truth[filename]
            pred_idx = predict_image(model, img_path)
            pred_class = idx_to_class[pred_idx]
            y_true.append(true_class)
            y_pred.append(pred_class)

    labels = sorted(set(y_true + y_pred))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for cls in labels:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    avg_precision = precision_sum / len(labels)
    avg_recall = recall_sum / len(labels)
    avg_f1 = f1_sum / len(labels)
    accuracy = correct / len(y_true)

    print("\n✅ Task B Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-score:  {avg_f1:.4f}")

def predict_only(model_path, test_folder, output_csv, label_csv_path):
    model = load_model(model_path, len(class_names), backbone="resnet34")
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")]
    image_files_sorted = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    predictions = []
    filenames = []

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])

        for filename in image_files_sorted:
            file_path = os.path.join(test_folder, filename)
            prediction = class_names[predict_image(model, file_path)]
            writer.writerow([filename, prediction])
            predictions.append(prediction)
            filenames.append(filename)

    print(f"\n✅ Predictions CSV saved to: {output_csv}")

    ground_truth = load_ground_truth(label_csv_path)
    y_true = [ground_truth[f] for f in filenames if f in ground_truth]
    y_pred = [pred for f, pred in zip(filenames, predictions) if f in ground_truth]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n✅ Task A Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        model_path = sys.argv[1]
        test_folder = sys.argv[2]
        label_csv_path = sys.argv[3]
        class_map_path = sys.argv[4]
        evaluate_with_gt(model_path, test_folder, label_csv_path, class_map_path)
    elif len(sys.argv) == 4:
        model_path = sys.argv[1]
        test_folder = sys.argv[2]
        label_csv_path = sys.argv[3]
        output_csv = os.path.join(test_folder, "predictions.csv")
        predict_only(model_path, test_folder, output_csv, label_csv_path)
    else:
        print("🔄 No command-line arguments detected. Switching to interactive input mode.")
        model_path = input("Enter the path to the model (.pth) file: ").strip()
        test_folder = input("Enter the path to the test images folder: ").strip()
        label_csv_path = input("Enter the path to the ground truth CSV file: ").strip()
        class_map_path = input("Enter the path to the class map JSON file (or leave blank for Task A): ").strip()

        output_csv = os.path.join(test_folder, "predictions.csv")
        if class_map_path:
            evaluate_with_gt(model_path, test_folder, label_csv_path, class_map_path)
        else:
            predict_only(model_path, test_folder, output_csv, label_csv_path)
