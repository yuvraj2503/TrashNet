# ♻️ Scrap Sorting ML System

**Real-time classification of recyclable materials (cardboard, glass, metal, paper, plastic, trash) using computer vision and deep learning.**

This project demonstrates how transfer learning and lightweight deployment can enable automated scrap sorting for industrial recycling systems.

---

## 🌍 Motivation

Recycling and waste management require accurate separation of materials. Manual sorting is:

- Labor-intensive  
- Error-prone  
- Expensive at scale  

Using deep learning, we can build automated systems capable of identifying and sorting scrap in real time, reducing cost and increasing efficiency.

---

## 🚀 Key Features

- **6-Class Classification:** Cardboard, Glass, Metal, Paper, Plastic, Trash  
- **Data Augmentation:** Improves generalization on limited datasets  
- **Transfer Learning:** ResNet18 pretrained on ImageNet, fine-tuned on TrashNet dataset  
- **Lightweight Deployment:** Exported to both TorchScript and ONNX for edge devices  
- **Conveyor Simulation:** Mimics real-time scrap sorting pipeline  
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix  

---

## 🗃️ Dataset

- **Source:** [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet)  
- **Classes:** Cardboard, Glass, Metal, Paper, Plastic, Trash  
- **Split:**  
  - 70% — Training  
  - 20% — Validation  
  - 10% — Testing  
- **Augmentations:** Random rotation, flips, color jittering  

---

## 🧠 Model & Training

| Component | Description |
|:-----------|:-------------|
| **Architecture** | ResNet18 |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | Adam (lr = 1e-4) |
| **Scheduler** | ReduceLROnPlateau |
| **Epochs** | 3 |
| **Hardware** | Google Colab GPU (T4) |

### 🔧 Exports
- PyTorch checkpoint (`.pth`)  
- TorchScript (`.pt`)  
- ONNX (`.onnx`)

---

## 📈 Results

### 🧮 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|:------------|-----------:|--------:|----------:|---------:|
| Cardboard  | 0.9740 | 1.0000 | 0.9868 | 75 |
| Glass      | 0.9091 | 0.8738 | 0.8911 | 103 |
| Metal      | 0.8462 | 0.8462 | 0.8462 | 78 |
| Paper      | 0.9569 | 0.9737 | 0.9652 | 114 |
| Plastic    | 0.9048 | 0.9048 | 0.9048 | 105 |
| Trash      | 0.8462 | 0.8462 | 0.8462 | 26 |

accuracy                         0.9162       501

| **Macro Avg** | 0.9062 | 0.9074 | 0.9067 | 501 |
| **Weighted Avg** | 0.9157 | 0.9162 | 0.9158 | 501 |


- **Overall Accuracy:** ~91.6%  
- **Strong performance** on cardboard, paper, and plastic.  
- **Slightly lower recall** for glass, metal, and trash due to class imbalance.

### 📊 Confusion Matrix
The confusion matrix is available at:  
`/results/confusion_matrix.png`

---

## 📂 Project Structure
TrashNet/
│
├── models/ # Trained model files and metadata
│ └── model_info.txt
│
├── performance/ # Performance analysis and reports
│ └── Performance_Report.md
│
├── results/ # Outputs and visualizations
│ ├── confusion_matrix.png
│ ├── conveyor_grid_100.png
│ └── predictions.csv
│
├── src/ # Source code and notebooks
│ ├── trash_net_script.ipynb
│ └── trash_net_script.py
│
├── LICENSE # License file
└── README.md # Project documentation


---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install torch torchvision datasets huggingface_hub tqdm scikit-learn onnxruntime pillow matplotlib
```
2️⃣ Train Model (optional)
```bash
jupyter notebook src/training_notebook.ipynb
```
### 3️⃣ Run Inference
```bash
python src/trash_net_script.py
```
4️⃣ Conveyor Simulation (Optional)
```bash
python src/trash_net_script.py --simulate
```
### 5️⃣ Check Results

- **CSV Logs:** `results/predictions.csv`
 
- **Confusion Matrix:** `results/confusion_matrix.png`
 
- **Visualization Grid:** `results/conveyor_grid_100.png`

---

## 📊 Performance Report

See [`performance/Performance_Report.md`](performance/Performance_Report.md) for:

- Training curves
 
- Confusion matrix
 
- Per-class precision/recall
 
- Key takeaways


## 🔧 Future Work

- Deploy on **Jetson Nano / Xavier** for edge inference
 
- Add **manual override logic** for misclassifications
 
- Implement **active learning** loop to retrain with misclassified samples


## 🙌 Credits

- **Dataset:** [garythung/TrashNet](https://huggingface.co/datasets/garythung/trashnet)
 
- **Frameworks:** PyTorch, Hugging Face Datasets, ONNX Runtime
