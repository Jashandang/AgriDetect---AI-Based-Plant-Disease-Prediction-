# 🌿 AgriDetect – Plant Disease Detection

> **Deep-learning powered, multilingual (EN/HI/PA) leaf disease classification** with a choice of models:
>
> * **Fusion**: ResNet-18 (features) + EfficientNet-B0 (features) → Fusion Head
> * **ResNet‑50**: Standalone classifier
>
> Includes a **Streamlit dashboard** (Colab-ready), **training/evaluation scripts**, and **actionable recommendations** for 15 PlantVillage classes in **English, Hindi, and Punjabi**.

---

## 🔗 Quick Links

* **Demo (Colab)**: *Run Streamlit on Colab and expose via localtunnel.*
* **Dataset**: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) 
* **Issues / Questions**: Please open a GitHub Issue.

---

## ✨ Features

* 📦 **Model options**: Fusion (ResNet18+EffB0) **or** ResNet‑50
* 🩺 **Multilingual recommendations**: English / हिंदी / ਪੰਜਾਬੀ
* ⚡ **Colab-first**: Minimal setup; works on CPU/GPU
* 🧠 **Transfer learning** on ImageNet backbones
* 🧱 **Clean modular code**: training, evaluation, Streamlit app
* 🔍 **Top‑k predictions** + confidence and runtime stats

---

## 🧬 Supported Classes (15)

```
Pepper__bell___Bacterial_spot, Pepper__bell___healthy,
Potato___Early_blight, Potato___Late_blight, Potato___healthy,
Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Late_blight,
Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot,
Tomato_Spider_mites_Two_spotted_spider_mite,
Tomato__Target_Spot, Tomato__Tomato_YellowLeaf__Curl_Virus,
Tomato__Tomato_mosaic_virus, Tomato_healthy
```

---

## 🏗️ Architecture Overview

**Fusion pipeline**

1. Image → **ResNet‑18** (remove `fc`) → feature vector *(512)*
2. Image → **EfficientNet‑B0** (remove `classifier`) → feature vector *(1280)*
3. `concat([512, 1280]) → 512 → 128 → num_classes` with BN + Dropout

**ResNet‑50 pipeline**

* Image → **ResNet‑50** (replace final `fc`) → logits *(num_classes)*

---

## 📊 Results (Illustrative)

> Your results will vary by training split and seed.

|                         Model |         Val Acc        |       Test Acc       |
| ----------------------------: | :--------------------: | :------------------: |
| ResNet‑18 (frozen + new head) |         ~92–93%        | **93.8%** (reported) |
|          Fusion (Res18+EffB0) | *typically ≥ ResNet18* |           —          |
|                     ResNet‑50 |   depends on training  |           —          |

> Tip: Unfreezing and fine‑tuning backbones after the head converges can boost accuracy.

---

## 📁 Repository Structure

```
.
├─ app.py                      # Streamlit dashboard (Colab-ready)
├─ train_resnet18.py           # Example training script (optional)
├─ train_efficientnet_b0.py    # Example training script (optional)
├─ train_resnet50.py           # Example training script (optional)
├─ fuse_train.py               # Fusion head training (optional)
├─ class_names.json            # 15 class names (created in Colab step)
├─ models/                     # (optional) place .pth files here or in /content on Colab
└─ assets/                     # screenshots, diagrams
```

---

## ⚙️ Setup (Colab)

1. **Mount Drive** and install dependencies

```python
from google.colab import drive
drive.mount('/content/drive')
!pip -q install streamlit==1.36.0 localtunnel torch torchvision torchaudio
```

2. **Create** `class_names.json` (if not already available)

```python
import json
classes = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]
with open('/content/class_names.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f)
print('Wrote /content/class_names.json')
```

3. **Place your trained weights** in `/content` or your Drive and update paths inside `app.py`:

```
RESNET18_PATH     = "/content/drive/MyDrive/AgriDetect/plant_disease_resnet18_best.pth"
EFFICIENTNET_PATH = "/content/drive/MyDrive/AgriDetect/plant_disease_efficientnet_b0_best.pth"
FUSION_MODEL_PATH = "/content/drive/MyDrive/AgriDetect/plant_disease_fusion_best.pth"
RESNET50_PATH     = "/content/drive/MyDrive/AgriDetect/plant_disease_resnet50_best.pth"
CLASS_NAMES_PATH  = "/content/class_names.json"
```

4. **Write and run the app**

```python
%%writefile app.py
# (paste the app.py from this repo – Fusion + ResNet‑50 selector)
```

```bash
# Run Streamlit and expose via localtunnel
streamlit run app.py & npx localtunnel --port 8501
```

> The terminal will print a public URL (from localtunnel). Open it to use the dashboard.

---

## 🖥️ Run Locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
streamlit run app.py
```

Update paths in `app.py` to point to your local `.pth` and `class_names.json`.

---

## 🧪 Training (outline)

> You can train any backbone you like. Below is a minimal outline.

**ResNet‑18 (feature extractor + new head)**

```python
from torchvision import models
import torch.nn as nn
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, num_classes)
```

Train on your `ImageFolder` dataset split: 70%/15%/15% is common. Save the best model:

```python
torch.save(model.state_dict(), 'plant_disease_resnet18_best.pth')
```

**EfficientNet‑B0**: same idea; replace `model.classifier[1]` with `nn.Linear(in_f, num_classes)` and save weights.

**Fusion Head**: build a small MLP that consumes `concat(res18_feat, effb0_feat)` and train it with both backbones frozen.

---

## 🗂️ Model Zoo 

Place these files in `/content` (Colab) or adjust paths:

```
plant_disease_resnet18_best.pth
plant_disease_efficientnet_b0_best.pth
plant_disease_fusion_best.pth
plant_disease_resnet50_best.pth
class_names.json
```

> The dashboard automatically detects which models are present and shows them in the selector.

---

## 🌐 Multilingual Recommendations (ENGLISH/HINDI/PUNJABI)

* For each predicted class, the app displays short, actionable guidance in **English**, **Hindi**, and **Punjabi**.
* Add or edit content in the `SOLUTIONS` dictionary inside `app.py`.

---

## 🧯 Troubleshooting

**Fusion unavailable in UI**

* One or more files missing. Verify `RESNET18_PATH`, `EFFICIENTNET_PATH`, `FUSION_MODEL_PATH`.
* Ensure the fusion head was trained with feature dims **512 (Res18)** and **1280 (EffB0)**.

**`Missing class_names.json`**

* Create it with the Colab snippet above and ensure `CLASS_NAMES_PATH` points to it.

**`PyTorchStreamReader failed reading zip archive`**

* The `.pth` is corrupted or not a valid PyTorch checkpoint (sometimes happens after Drive download). Re‑upload or re‑save the file.

**`__init__() takes 1 positional argument but 4 were given`**

* Caused by a malformed class definition. Use the provided `FusionNet` implementation.

**GPU not available on Colab**

* Switch to CPU (works fine, just slower) or try again later. Paid tiers provide better GPU availability.

---

## 🚀 Roadmap

* [ ] Optional Grad‑CAM visualizations
* [ ] Batch inference & CSV export
* [ ] On‑device (TFLite/CoreML) model export
* [ ] Fine‑tuning and mixed‑precision training scripts

---

## 🤝 Contributing

PRs are welcome! Please:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit with clear messages
4. Open a Pull Request with context and screenshots

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

* **PlantVillage** dataset and community
* PyTorch and TorchVision teams
* Streamlit for rapid UI

---

## 📣 Citation

If you use this project, please cite it as:

```
Tanveer et al., "AgriDetect: Multilingual Plant Disease Detection with Fusion of ResNet‑18 and EfficientNet‑B0," 2025. GitHub repository.
```

---

## 🖼️ Screenshots

> Add your own screenshots under `assets/` and reference them here.

![Dashboard](assets/dashboard.png)
![Top‑3 Predictions](assets/top3.png)
