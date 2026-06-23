# 🌿 Plant Disease Detection with Explainable AI

An end-to-end web app that detects plant diseases from leaf images, explains *why* it made that decision using Grad-CAM, and gives practical treatment advice through an AI-powered "Plant Doctor."

> Upload a photo of a leaf → get a diagnosis across 38 plant disease/healthy categories → see exactly which part of the leaf the model focused on → get plain-language advice on what to do next.

## 🚀 Demo
<img width="547" height="371" alt="Screenshot 2026-06-19 161542" src="https://github.com/user-attachments/assets/34dbddd1-61f2-42ca-85a3-1434f11ae93d" />
[Uploading LICENSE…]()
<img width="667" height="366" alt="Screenshot 2026-06-19 161643" src="https://github.com/user-attachments/assets/af6414e8-a345-4bbf-86d7-3965cd2bfc56" />
<img width="590" height="260" alt="Screenshot 2026-06-19 161605" src="https://github.com/user-attachments/assets/44066cbb-7ef9-454e-91b0-10e2af0688f8" />


*(Add a screenshot or screen recording here — this is the single best thing you can add to make the repo stand out. Drag an image/GIF into this README via GitHub's editor and it'll embed automatically.)*

## ✨ Features

- **Disease Classification** — Identifies 38 plant disease/healthy categories across crops like Tomato, Apple, Grape, Corn, Potato, and more, with a confidence score for each prediction
- **Grad-CAM Explainability** — Generates a heatmap showing exactly which regions of the leaf influenced the model's decision, instead of returning a black-box prediction
- **AI Plant Doctor** — Uses an LLM (Llama 3.3 via Groq) to translate the raw prediction into a clear explanation: what the disease is, its symptoms, causes, treatment, and prevention — or practical care tips if the leaf is healthy
- **Fast & Lightweight** — Built on a transfer-learned MobileNetV2 backbone, making it fast enough to run interactively in a browser

## 🧠 How It Works

1. **Image preprocessing** — Uploaded leaf image is resized to 224×224 and normalized
2. **Classification** — A MobileNetV2 model (ImageNet-pretrained convolutional base, frozen, with a custom trained classification head) predicts the disease class
3. **Explainability** — Grad-CAM computes which spatial regions of the last convolutional layer most influenced the predicted class, producing a heatmap overlay
4. **AI explanation** — The predicted class name is passed to an LLM, which generates a farmer-friendly explanation tailored to that specific disease (or health status)

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model | TensorFlow / Keras, MobileNetV2 (transfer learning) |
| Explainability | Grad-CAM (custom implementation) |
| AI Doctor | Groq API — `llama-3.3-70b-versatile` |
| Web App | Streamlit |
| Image Processing | OpenCV |

## 📋 Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com) (no credit card required)

## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/fariskonnengal/plant_disease_detection.git
   cd plant_disease_detection
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit tensorflow opencv-python matplotlib groq
   ```

3. **Add your Groq API key**

   Create a file at `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your-api-key-here"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`) and upload a leaf image.

## 📁 Project Structure

```
plant_disease_detection/
├── app.py                      # Main Streamlit application
├── plant_disease_model.keras   # Trained MobileNetV2 model
├── .streamlit/
│   └── secrets.toml            # API key (not committed — see .gitignore)
└── README.md
```

## 🧩 Supported Classes

Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato — covering both diseased and healthy leaf states for each, totaling 38 classes.

## ⚠️ Known Limitations

- **Grad-CAM on healthy predictions** is less interpretable than on diseased ones — since there's no localized symptom to point to, the heatmap reflects general texture cues rather than a specific defect
- **Groq's free tier** has rate limits (~100–150 AI Doctor explanations/day with this model) — plenty for demos, but would need a paid tier for production-scale usage
- The model's convolutional base is frozen (transfer learning only) — it has not been fine-tuned specifically on plant disease imagery, which keeps training fast and lightweight but means feature extraction relies entirely on ImageNet-pretrained features
