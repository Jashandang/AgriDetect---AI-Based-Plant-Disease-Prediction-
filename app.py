# ===============================================================
# 🌿 Plant Disease Detection Dashboard (Colab-ready)
# Choose: Fusion (ResNet18 + EfficientNetB0)  OR  ResNet-50
# EN / HI / PA recommendations for 15 classes
# ===============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json, os, time
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🌿 Plant Disease Detector",
                   page_icon="🌿", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# speed tweaks
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    except Exception:
        pass

# === Update these paths if needed (use exact filenames in /content) ===
RESNET18_PATH     = "/content/plant_disease_resnet18_best.pth"            # backbone for Fusion
EFFICIENTNET_PATH = "/content/plant_disease_efficientnet_b0_best.pth"      # backbone for Fusion
FUSION_MODEL_PATH = "/content/plant_disease_fusion_best.pth"           # Fusion head (change if your file has no (1))
RESNET50_PATH     = "/content/plant_disease_resnet50_best.pth"             # standalone ResNet-50
CLASS_NAMES_PATH  = "/content/class_names.json"                             # same list used in training


# ---------------- SOLUTIONS (EN/HI/PA) ----------------
SOLUTIONS = {
    "Pepper__bell___Bacterial_spot": {
        "en": """**Bacterial Spot (Pepper) — Actions**
- Remove & destroy infected leaves/fruits.
- Spray copper-based bactericide as per label.
- Avoid working in wet foliage.

**Prevent:** disease-free seed/transplants, base irrigation, spacing & crop rotation.""",
        "hi": """**बैक्टेरियल स्पॉट (शिमला मिर्च) — उपाय**
- संक्रमित पत्ते/फल हटाकर नष्ट करें।
- लेबल अनुसार कॉपर आधारित बैक्टीरिसाइड स्प्रे करें।
- गीले पत्तों में काम न करें।

**रोकथाम:** रोग-मुक्त बीज/नर्सरी, ड्रिप सिंचाई, दूरी रखें, फसल चक्र अपनाएँ।""",
        "pa": """**ਬੈਕਟੀਰੀਅਲ ਸਪੌਟ (ਸ਼ਿਮਲਾ ਮਿਰਚ) — ਹੱਲ**
- ਸੰਕ੍ਰਮਿਤ ਪੱਤੇ/ਫਲ ਹਟਾ ਕੇ ਨਸ਼ਟ ਕਰੋ।
- ਲੇਬਲ ਮੁਤਾਬਕ ਤਾਂਬੇ-ਆਧਾਰਿਤ ਸਪਰੇ ਕਰੋ।
- ਗਿੱਲੇ ਪੱਤਿਆਂ 'ਤੇ ਕੰਮ ਨਾ ਕਰੋ।

**ਰੋਕਥਾਮ:** ਰੋਗ-ਰਹਿਤ ਬੀਜ/ਟ੍ਰਾਂਸਪਲਾਂਟ, ਬੇਸ ਸਿੰਚਾਈ, ਫਾਸਲਾ, ਫਸਲ ਘੁਮਾਵ।"""
    },
    "Pepper__bell___healthy": {
        "en": "Healthy pepper — no treatment needed. Prevent with scouting, balanced fertilization, base irrigation & rotation.",
        "hi": "स्वस्थ शिमला मिर्च — उपचार आवश्यक नहीं। निरीक्षण, संतुलित खाद, आधार सिंचाई व फसल चक्र रखें।",
        "pa": "ਸਿਹਤਮੰਦ ਸ਼ਿਮਲਾ ਮਿਰਚ — ਇਲਾਜ ਲੋੜੀਂਦਾ ਨਹੀਂ। ਨਿਯਮਤ ਜਾਂਚ, ਸੰਤੁਲਿਤ ਖਾਦ, ਬੇਸ ਸਿੰਚਾਈ ਅਤੇ ਫਸਲ ਘੁਮਾਵ।"
    },
    "Potato___Early_blight": {
        "en": """**Early Blight (Potato) — Actions**
- Remove infected leaves.
- Fungicide: chlorothalonil / mancozeb (per label).
- Mulch to reduce soil splash.

**Prevent:** avoid overhead irrigation; rotate; use resistant varieties.""",
        "hi": """**अर्ली ब्लाइट (आलू) — उपाय**
- संक्रमित पत्ते हटाएँ।
- क्लोरोथैलोनिल/मैनकोजेब फफूंदनाशक (लेबल अनुसार)।
- मल्च डालें ताकि छींटे कम हों।

**रोकथाम:** ऊपर से सिंचाई से बचें, फसल चक्र, प्रतिरोधी किस्में।""",
        "pa": """**ਅਰਲੀ ਬਲਾਈਟ (ਆਲੂ) — ਹੱਲ**
- ਸੰਕ੍ਰਮਿਤ ਪੱਤੇ ਹਟਾਓ।
- ਕਲੋਰੋਥੈਲੋਨਿਲ/ਮੈਨਕੋਜ਼ੇਬ ਫੰਗੀਸਾਈਡ (ਲੇਬਲ ਮੁਤਾਬਕ)।
- ਮਲਚ ਨਾਲ ਛਿਟਕਣ ਘਟਾਓ।

**ਰੋਕਥਾਮ:** ਓਵਰਹੈੱਡ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ, ਫਸਲ ਘੁਮਾਵ, ਰੋਧਕ ਕਿਸਮਾਂ।"""
    },
    "Potato___Late_blight": {
        "en": """**Late Blight (Potato) — Actions**
- Uproot & destroy infected plants.
- Copper-based fungicides per local guidelines.
- Ensure drainage.

**Prevent:** avoid overhead irrigation; certified seed; rotation.""",
        "hi": """**लेट ब्लाइट (आलू) — उपाय**
- संक्रमित पौधे उखाड़कर नष्ट करें।
- स्थानीय दिशा-निर्देश अनुसार कॉपर फफूंदनाशक।
- जल निकासी ठीक रखें।

**रोकथाम:** ऊपर से सिंचाई न करें; प्रमाणित बीज; फसल चक्र।""",
        "pa": """**ਲੇਟ ਬਲਾਈਟ (ਆਲੂ) — ਹੱਲ**
- ਸੰਕ੍ਰਮਿਤ ਪੌਦੇ ਹਟਾ ਕੇ ਨਸ਼ਟ ਕਰੋ।
- ਸਥਾਨਕ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼ ਅਨੁਸਾਰ ਤਾਂਬੇ ਵਾਲੇ ਫੰਗੀਸਾਈਡ।
- ਡਰੇਨੇਜ ਯਕੀਨੀ ਬਣਾਓ।

**ਰੋਕਥਾਮ:** ਓਵਰਹੈੱਡ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ; ਸਰਟੀਫਾਇਡ ਬੀਜ; ਫਸਲ ਘੁਮਾਵ।"""
    },
    "Potato___healthy": {
        "en": "Healthy potato — no treatment. Prevent with rotation, certified seed, good drainage, balanced nutrition.",
        "hi": "स्वस्थ आलू — उपचार नहीं। रोकथाम: फसल चक्र, प्रमाणित बीज, जल निकासी, संतुलित पोषण।",
        "pa": "ਸਿਹਤਮੰਦ ਆਲੂ — ਇਲਾਜ ਨਹੀਂ। ਰੋਕਥਾਮ: ਫਸਲ ਘੁਮਾਵ, ਸਰਟੀਫਾਇਡ ਬੀਜ, ਡਰੇਨੇਜ, ਸੰਤੁਲਿਤ ਪੋਸ਼ਣ।"
    },
    "Tomato_Bacterial_spot": {
        "en": """**Bacterial Spot (Tomato) — Actions**
- Remove infected leaves; prune for airflow.
- Copper sprays as recommended.
- Sanitize tools.

**Prevent:** disease-free transplants; avoid overhead irrigation.""",
        "hi": """**बैक्टीरियल स्पॉट (टमाटर) — उपाय**
- संक्रमित पत्ते हटाएँ; हवा के लिए छंटाई करें।
- अनुशंसित कॉपर स्प्रे।
- औज़ारों की सफाई रखें।

**रोकथाम:** रोग-रहित पौध; ऊपर से सिंचाई न करें।""",
        "pa": """**ਬੈਕਟੀਰੀਅਲ ਸਪੌਟ (ਟਮਾਟਰ) — ਹੱਲ**
- ਸੰਕ੍ਰਮਿਤ ਪੱਤੇ ਹਟਾਓ; ਹਵਾ ਲਈ ਪ੍ਰੂਨ ਕਰੋ।
- ਤਾਮਬੇ ਦੇ ਸਪਰੇ।
- ਸੰਦ ਸੈਨੀਟਾਈਜ਼ ਕਰੋ।

**ਰੋਕਥਾਮ:** ਰੋਗ-ਰਹਿਤ ਪੌਦੇ; ਓਵਰਹੈੱਡ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ।"""
    },
    "Tomato_Early_blight": {
        "en": "Remove infected parts; spray chlorothalonil/mancozeb; mulch; rotate crops.",
        "hi": "संक्रमित हिस्से हटाएँ; क्लोरोथैलोनिल/मैनकोजेब स्प्रे; मल्च; फसल चक्र अपनाएँ।",
        "pa": "ਸੰਕ੍ਰਮਿਤ ਹਿੱਸੇ ਹਟਾਓ; ਕਲੋਰੋਥੈਲੋਨਿਲ/ਮੈਨਕੋਜ਼ੇਬ; ਮਲਚ; ਫਸਲ ਬਦਲੋ।"
    },
    "Tomato_Late_blight": {
        "en": "Remove infected plants; copper fungicides; good drainage; avoid overhead irrigation.",
        "hi": "संक्रमित पौधे हटा दें; कॉपर फफूंदनाशक; अच्छी निकासी; ऊपर से सिंचाई न करें।",
        "pa": "ਸੰਕ੍ਰਮਿਤ ਪੌਦੇ ਹਟਾਓ; ਤਾਂਬੇ ਵਾਲੇ ਫੰਗੀਸਾਈਡ; ਚੰਗੀ ਡਰੇਨੇਜ; ਓਵਰਹੈੱਡ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ।"
    },
    "Tomato_Leaf_Mold": {
        "en": "Improve airflow; remove lower leaves; recommended fungicide; avoid high humidity irrigation.",
        "hi": "हवा का प्रवाह बढ़ाएँ; निचले पत्ते हटाएँ; अनुशंसित फफूंदनाशक; अधिक आर्द्र सिंचाई से बचें।",
        "pa": "ਹਵਾਦਾਰੀ ਵਧਾਓ; ਹੇਠਲੇ ਪੱਤੇ ਹਟਾਓ; ਫੰਗੀਸਾਈਡ; ਨਮੀ ਵਾਲੀ ਓਵਰਹੈੱਡ ਸਿੰਚਾਈ ਤੋਂ ਬਚੋ।"
    },
    "Tomato_Septoria_leaf_spot": {
        "en": "Remove infected leaves; apply suitable fungicide; mulch; avoid splash irrigation.",
        "hi": "संक्रमित पत्ते हटाएँ; उचित फफूंदनाशक; मल्च; छींटे वाली सिंचाई से बचें।",
        "pa": "ਸੰਕ੍ਰਮਿਤ ਪੱਤੇ ਹਟਾਓ; ਢੁੱਕਵਾਂ ਫੰਗੀਸਾਈਡ; ਮਲਚ; ਛਿਡਕਾਅ ਤੋਂ ਬਚੋ।"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "en": "Spray water to dislodge; use miticides if severe; encourage predators (ladybugs); reduce water-stress.",
        "hi": "पानी का स्प्रे करें; गंभीर होने पर माइटिसाइड; लेडीबग जैसे शिकारियों को बढ़ावा दें; जल-तनाव कम करें।",
        "pa": "ਪਾਣੀ ਦਾ ਸਪਰੇ; ਗੰਭੀਰ ਹੋਣ 'ਤੇ ਮਾਈਟਿਸਾਈਡ; ਲੇਡੀਬੱਗ ਵਰਗੇ ਸ਼ਿਕਾਰੀ; ਪਾਣੀ-ਤਣਾਅ ਘਟਾਓ।"
    },
    "Tomato__Target_Spot": {
        "en": "Remove infected tissue; rotate fungicides; avoid long leaf wetness; crop rotation.",
        "hi": "संक्रमित ऊतक हटाएँ; फफूंदनाशकों का परिवर्तन; पत्तों को लंबे समय तक गीला न रखें; फसल चक्र।",
        "pa": "ਸੰਕ੍ਰਮਿਤ ਹਿੱਸੇ ਹਟਾਓ; ਫੰਗੀਸਾਈਡ ਘੁਮਾਓ; ਲੰਬੀ ਨਮੀ ਤੋਂ ਬਚੋ; ਫਸਲ ਘੁਮਾਵ।"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "en": "Remove infected plants; control whiteflies (insecticidal soap/Neem/reflective mulch); use resistant varieties.",
        "hi": "संक्रमित पौधे हटाएँ; व्हाइटफ्लाई नियंत्रण (इन्सेक्टिसाइडल सोप/नीम/रिफ्लेक्टिव मल्च); प्रतिरोधी किस्में अपनाएँ।",
        "pa": "ਪੌਦੇ ਹਟਾਓ; ਵ੍ਹਾਈਟਫਲਾਈ ਕੰਟਰੋਲ (ਸਾਬਣ/ਨੀਮ/ਰਿਫਲੈਕਟਿਵ ਮਲਚ); ਰੋਧਕ ਕਿਸਮਾਂ ਵਰਤੋ।"
    },
    "Tomato__Tomato_mosaic_virus": {
        "en": "Remove infected plants; sanitize tools; avoid tobacco near plants; use virus-free seedlings.",
        "hi": "संक्रमित पौधे हटाएँ; औज़ार साफ रखें; पौधों के पास तम्बाकू से बचें; वायरस-रहित पौध लगाएँ।",
        "pa": "ਪੌਦੇ ਹਟਾਓ; ਸੰਦ ਸਾਫ ਰੱਖੋ; ਤਮਾਕੂ ਤੋਂ ਬਚੋ; ਵਾਇਰਸ-ਫ੍ਰੀ ਪੌਦੇ ਲਗਾਓ।"
    },
    "Tomato_healthy": {
        "en": "Healthy tomato — no treatment. Prevent with balanced nutrition, scouting, sanitation & proper irrigation.",
        "hi": "स्वस्थ टमाटर — उपचार नहीं। रोकथाम: संतुलित पोषण, निरीक्षण, स्वच्छता, उचित सिंचाई।",
        "pa": "ਸਿਹਤਮੰਦ ਟਮਾਟਰ — ਇਲਾਜ ਨਹੀਂ। ਰੋਕਥਾਮ: ਸੰਤੁਲਿਤ ਖੁਰਾਕ, ਜਾਂਚ, ਸਫਾਈ, ਠੀਕ ਸਿੰਚਾਈ।"
    }
}
# ---------------- END SOLUTIONS ----------------


# ---------------- MODEL CLASSES & LOADERS ----------------
class FusionNet(nn.Module):
    def __init__(self, resnet_dim: int, effnet_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(resnet_dim + effnet_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x_res, x_eff):
        x = torch.cat((x_res, x_eff), dim=1)
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


@st.cache_resource
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Missing {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_fusion_models(class_names):
    needed = [RESNET18_PATH, EFFICIENTNET_PATH, FUSION_MODEL_PATH]
    if any(not os.path.exists(p) for p in needed):
        st.warning("Fusion unavailable: missing one or more files.")
        return None, None, None

    try:
        # ResNet-18 backbone
        res18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        res18.fc = nn.Identity()
        res18.load_state_dict(torch.load(RESNET18_PATH, map_location=DEVICE), strict=False)
        res18.to(DEVICE).eval()

        # EfficientNet-B0 backbone
        effb0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        effb0.classifier = nn.Identity()
        effb0.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE), strict=False)
        effb0.to(DEVICE).eval()

        # Fusion head
        num_classes = len(class_names)
        fusion_head = FusionNet(512, 1280, num_classes)
        fusion_head.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE), strict=True)
        fusion_head.to(DEVICE).eval()

        # Sanity check (optional)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            rf = res18(dummy); ef = effb0(dummy)
            if rf.dim() > 2: rf = torch.flatten(rf, 1)
            if ef.dim() > 2: ef = torch.flatten(ef, 1)
            _ = fusion_head(rf, ef)

        return res18, effb0, fusion_head

    except Exception as e:
        st.warning(f"Fusion unavailable: {e}")
        return None, None, None


@st.cache_resource
def load_resnet50_model(class_names):
    if not os.path.exists(RESNET50_PATH):
        st.warning("ResNet-50 weights not found; it will be unavailable.")
        return None
    try:
        num_classes = len(class_names)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        state = torch.load(RESNET50_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.warning(f"ResNet-50 unavailable: {e}")
        return None


# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------------- LOAD ALL ----------------
try:
    CLASS_NAMES = load_class_names()
except Exception as e:
    st.error("Failed to load class names.")
    st.exception(e)
    st.stop()

res18, effb0, fusion_head = load_fusion_models(CLASS_NAMES)
resnet50 = load_resnet50_model(CLASS_NAMES)

# Build available model options based on available files
model_options = []
if fusion_head is not None:
    model_options.append("Fusion (ResNet18 + EfficientNetB0)")
if resnet50 is not None:
    model_options.append("ResNet-50")

if not model_options:
    st.error("No models available. Check model paths at the top of app.py.")
    st.stop()


# ---------------- UI ----------------
st.title("🌱 Plant Disease Detection Dashboard")
st.caption(f"🖥️ Running on: **{DEVICE.upper()}**")
st.write("Upload a leaf image to identify the plant disease and get recommended actions in **English / Hindi / Punjabi**.")
st.divider()

model_choice = st.radio("Select model for prediction:", model_options, horizontal=True)

uploaded_file = st.file_uploader("Upload an image of a plant leaf 🍃", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True, caption="Uploaded image")

    lang = st.selectbox("Choose language / भाषा चुनें / ਭਾਸ਼ਾ ਚੁਣੋ", ["English","Hindi","Punjabi"])
    lang_map = {"English":"en","Hindi":"hi","Punjabi":"pa"}
    selected_lang = lang_map[lang]

    with st.spinner("Analyzing..."):
        x = transform(image).unsqueeze(0).to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            if model_choice.startswith("Fusion"):
                rf = res18(x)
                ef = effb0(x)
                if rf.dim() > 2: rf = torch.flatten(rf, 1)
                if ef.dim() > 2: ef = torch.flatten(ef, 1)
                logits = fusion_head(rf, ef)
            else:  # ResNet-50
                logits = resnet50(x)
        infer_ms = (time.time() - t0) * 1000.0

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        top3 = np.argsort(probs)[::-1][:3]

    st.markdown("### 🌿 Prediction Results")
    st.caption(f"⏱️ Inference time: {infer_ms:.1f} ms")
    st.markdown(f"**Predicted Class:** `{CLASS_NAMES[top_idx]}`")
    st.progress(float(probs[top_idx]))
    st.write(f"**Confidence:** {probs[top_idx]*100:.2f}%")

    st.markdown("### 🔝 Top-3 Predictions")
    for i in top3:
        st.write(f"- **{CLASS_NAMES[i]}** — {probs[i]*100:.2f}%")

    # Solutions
    predicted_key = CLASS_NAMES[top_idx]
    sol_dict = SOLUTIONS.get(predicted_key, {})
    solution_text = sol_dict.get(selected_lang, None)

    st.markdown("### 🩺 Recommended Actions / उपचार / ਸਲਾਹ")
    if solution_text:
        st.markdown(solution_text)
    else:
        st.warning("Solution not found for this class/language. "
                   "If you changed class names, update the SOLUTIONS keys accordingly.")
        with st.expander("Debug: Expected keys vs Predicted"):
            st.write("Predicted key:", predicted_key)
            st.write("Have solutions for keys:", list(SOLUTIONS.keys()))

else:
    st.info("👆 Upload an image to get prediction and solutions (EN/HI/PA).")

# ---------------- Sidebar ----------------
st.sidebar.title("About")
about = []
if fusion_head is not None:
    about.append("• **Fusion** = ResNet-18 (features) + EfficientNet-B0 (features) → Fusion Head")
if resnet50 is not None:
    about.append("• **ResNet-50** = standalone classifier")
st.sidebar.write("\n".join(about))
st.sidebar.write("Ensure `.pth` files and `class_names.json` exist at the paths defined at the top.")
