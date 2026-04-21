"""
app/streamlit_app.py
====================
Streamlit deployment app for Multimodal Negative Emotion Recognition.

Features
--------
  ✅ Upload EEG CSV  OR  use simulated EEG signal
  ✅ Upload face image  OR  use webcam
  ✅ Real-time face detection
  ✅ Emotion probability bar chart
  ✅ Temporal attention heatmap
  ✅ Emotion trend history (rolling window)
  ✅ Mental health alert system
  ✅ Model comparison tab

Run
---
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import cv2
from PIL import Image

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root on path ──────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.face_pipeline import FaceDetector, frame_to_tensor
from src.data.eeg_pipeline import EEGPipeline
from src.evaluation.explainability import GradCAMVisualizer, plot_eeg_attention
from src.data.dataset import simulate_data

# ── Constants ─────────────────────────────────────────────────
EMOTION_LABELS = ["anger", "fear", "sadness", "disgust"]
EMOTION_COLORS = {
    "anger":   "#EF4444",
    "fear":    "#F59E0B",
    "sadness": "#3B82F6",
    "disgust": "#8B5CF6",
}
ALERT_THRESHOLD = 0.70   # confidence above which we show mental health alert
TREND_WINDOW    = 30     # rolling window for trend graph

CHECKPOINT = "checkpoints/multimodal_best.pt"

# ─────────────────────────────────────────────────────────────
# Cached resource loading
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> Optional[MultimodalEmotionModel]:
    """Load checkpoint if it exists, otherwise return untrained model."""
    model = MultimodalEmotionModel()
    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        st.sidebar.success("✅ Checkpoint loaded")
    else:
        st.sidebar.warning("⚠️ No checkpoint found — using random weights.")
    model.eval()
    return model


@st.cache_resource
def load_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource
def load_eeg_pipeline() -> EEGPipeline:
    pipeline = EEGPipeline(seq_len=128, n_features=32)
    target_cols = pipeline.seq_len * pipeline.n_features
    # Fit the scaler with a neutral placeholder so inference-time transform works
    # even when the app is used without a prior training run.
    pipeline._scaler.fit(np.zeros((1, target_cols), dtype=np.float32))
    pipeline._fitted = True
    return pipeline


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=TREND_WINDOW)

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []


# ─────────────────────────────────────────────────────────────
# Helper: EEG tensor from uploaded CSV or simulation
# ─────────────────────────────────────────────────────────────

def get_eeg_tensor(
    uploaded_file,
    pipeline: EEGPipeline,
) -> torch.Tensor:
    """Return a (1, seq_len, n_features) EEG tensor."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        X_raw = df.values.astype(np.float32)
        # Take first row as the single sample
        row = X_raw[0:1] if X_raw.ndim == 2 else X_raw[np.newaxis, :]
        target_cols = pipeline.seq_len * pipeline.n_features
        if row.shape[1] < target_cols:
            row = np.pad(row, ((0, 0), (0, target_cols - row.shape[1])))
        else:
            row = row[:, :target_cols]
        X = pipeline.transform(row)        # (1, seq_len, n_feat)
    else:
        # Simulate a plausible EEG-like signal
        X, _ = simulate_data(n_samples=1, seq_len=128, n_features=32)

    return torch.tensor(X, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Helper: Face tensor from PIL Image
# ─────────────────────────────────────────────────────────────

def get_face_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Detect face, crop, and return model-ready tensor."""
    detector = load_face_detector()
    frame    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crops    = detector.detect_and_crop(frame)

    if crops:
        face_pil = crops[0]
    else:
        # Fallback: use the whole image
        face_pil = pil_img.convert("L")

    return frame_to_tensor(face_pil, img_size=48)   # (1, 1, 48, 48)


# ─────────────────────────────────────────────────────────────
# Alert logic
# ─────────────────────────────────────────────────────────────

def check_mental_health_alert(
    emotion: str,
    confidence: float,
    history: deque,
) -> Optional[str]:
    """
    Fire an alert when:
    - Confidence > threshold for a negative emotion, OR
    - The same emotion dominates > 70% of recent history window
    """
    if confidence >= ALERT_THRESHOLD:
        return (
            f"🚨 High confidence **{emotion}** detected ({confidence:.0%}). "
            "If you are experiencing distress, please reach out to a mental health professional."
        )

    if len(history) >= 10:
        recent = [e for e in history]
        dominant_frac = recent.count(emotion) / len(recent)
        if dominant_frac > 0.7:
            return (
                f"⚠️ Sustained **{emotion}** detected in {dominant_frac:.0%} of recent readings. "
                "Consider taking a break or speaking to someone you trust."
            )
    return None


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/EEG_cap.jpg/220px-EEG_cap.jpg",
        width=160,
        caption="EEG Emotion Recognition",
    )
    st.sidebar.title("⚙️ Settings")
    show_attention = st.sidebar.checkbox("Show EEG attention map", value=True)
    show_gradcam   = st.sidebar.checkbox("Show Grad-CAM heatmap",  value=False)
    show_trend     = st.sidebar.checkbox("Show emotion trend",      value=True)
    realtime_mode  = st.sidebar.radio("Input mode", ["Upload", "Webcam"])

    # ── Load resources ────────────────────────────────────────
    model    = load_model()
    pipeline = load_eeg_pipeline()
    pipeline._fitted = True  # allow transform without fit

    # ── Header ────────────────────────────────────────────────
    st.title("🧠 Multimodal Negative Emotion Recognition")
    st.markdown(
        "Uses **EEG signals** (processed by BiLSTM) and **facial images** "
        "(processed by CNN) with cross-modal attention fusion to classify: "
        "**Anger · Fear · Sadness · Disgust**"
    )
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    tab_predict, tab_compare, tab_about = st.tabs(
        ["🔮 Predict", "📊 Model Comparison", "ℹ️ About"]
    )

    # ─────────────────────────────────────────────────────────
    # TAB 1: Predict
    # ─────────────────────────────────────────────────────────
    with tab_predict:
        col_left, col_right = st.columns([1, 1], gap="large")

        # ── Left: Inputs ──────────────────────────────────────
        with col_left:
            st.subheader("📥 Inputs")

            # EEG input
            st.markdown("**EEG Signal**")
            eeg_file = st.file_uploader(
                "Upload EEG CSV (optional — simulated if blank)",
                type=["csv"],
                key="eeg_upload",
            )

            # Face input
            st.markdown("**Facial Image**")
            if realtime_mode == "Upload":
                face_file = st.file_uploader(
                    "Upload face image", type=["jpg", "jpeg", "png"],
                    key="face_upload"
                )
                pil_img = Image.open(face_file) if face_file else None
            else:
                cam_img = st.camera_input("Take a photo")
                pil_img = Image.open(cam_img) if cam_img else None

            run_btn = st.button("▶ Predict Emotion", use_container_width=True,
                                type="primary")

        # ── Right: Results ────────────────────────────────────
        with col_right:
            st.subheader("🎯 Prediction")

            if run_btn:
                if pil_img is None:
                    st.warning("Please provide a facial image.")
                else:
                    with st.spinner("Running inference…"):
                        try:
                            # ── Build tensors ──────────────────
                            eeg_tensor  = get_eeg_tensor(eeg_file, pipeline)
                            face_tensor = get_face_tensor(pil_img)

                            # ── Predict ────────────────────────
                            result = model.predict(face_tensor, eeg_tensor)

                            emotion    = result["label"]
                            confidence = result["confidence"]
                            probs      = result["probs"]
                            attn_w     = result["eeg_attn"]

                            # ── Show face ──────────────────────
                            detector = load_face_detector()
                            frame    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            boxes    = detector.detect_faces(frame)
                            annotated = detector.draw_boxes(frame, [emotion.upper()])
                            st.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                caption="Face Detection + Prediction",
                                use_container_width=True,
                            )

                            # ── Emotion badge ──────────────────
                            color = EMOTION_COLORS.get(emotion, "#6B7280")
                            st.markdown(
                                f"""
                                <div style="
                                    background:{color}22;
                                    border:2px solid {color};
                                    border-radius:12px;
                                    padding:16px;
                                    text-align:center;
                                    margin:8px 0;
                                ">
                                <span style="font-size:2rem;">
                                    {"😡" if emotion=="anger" else
                                     "😨" if emotion=="fear"  else
                                     "😢" if emotion=="sadness" else "🤢"}
                                </span><br>
                                <strong style="font-size:1.4rem;color:{color};">
                                    {emotion.upper()}
                                </strong><br>
                                <span style="color:#6B7280;">
                                    Confidence: {confidence:.1%}
                                </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # ── Probability bars ───────────────
                            st.markdown("**Class Probabilities**")
                            for lbl, prob in sorted(probs.items(),
                                                     key=lambda x: -x[1]):
                                col_lbl, col_bar = st.columns([2, 5])
                                with col_lbl:
                                    st.write(lbl.capitalize())
                                with col_bar:
                                    st.progress(float(prob),
                                                text=f"{prob:.1%}")

                            # ── EEG attention ──────────────────
                            if show_attention:
                                st.markdown("**EEG Temporal Attention**")
                                fig_path = "results/eeg_attn_tmp.png"
                                plot_eeg_attention(
                                    attn_w,
                                    emotion_label=emotion,
                                    save_path=fig_path,
                                )
                                st.image(fig_path,
                                         use_container_width=True)

                            # ── Update history ─────────────────
                            st.session_state.emotion_history.append(emotion)

                            # ── Alert ──────────────────────────
                            alert = check_mental_health_alert(
                                emotion, confidence,
                                st.session_state.emotion_history
                            )
                            if alert:
                                st.error(alert)
                                st.session_state.alert_log.append(
                                    {"time": time.strftime("%H:%M:%S"),
                                     "alert": alert}
                                )

                        except Exception as e:
                            st.error(f"Inference error: {e}")

        # ── Trend chart ───────────────────────────────────────
        if show_trend and len(st.session_state.emotion_history) > 1:
            st.divider()
            st.subheader("📈 Emotion Trend (Last 30 Readings)")
            history = list(st.session_state.emotion_history)
            df_trend = pd.DataFrame({
                "index": range(len(history)),
                "emotion": history,
            })
            counts = df_trend["emotion"].value_counts()
            st.bar_chart(counts, color=["#3B82F6"])

        # ── Alert log ─────────────────────────────────────────
        if st.session_state.alert_log:
            with st.expander("🔔 Alert History"):
                for a in reversed(st.session_state.alert_log[-10:]):
                    st.markdown(f"`{a['time']}` — {a['alert']}")

    # ─────────────────────────────────────────────────────────
    # TAB 2: Model Comparison
    # ─────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("📊 Model Performance Comparison")
        st.info("Results based on the research paper (Group 14).")

        comparison_data = {
            "Model":       ["CNN (Face only)", "BiLSTM (EEG only)", "Multimodal CNN+LSTM"],
            "Accuracy":    [0.82, 0.65, 0.89],
            "Macro F1":    [0.80, 0.63, 0.88],
            "Precision":   [0.81, 0.64, 0.87],
            "Recall":      [0.80, 0.62, 0.88],
        }
        df_cmp = pd.DataFrame(comparison_data).set_index("Model")
        st.dataframe(df_cmp.style.highlight_max(axis=0, color="#DCFCE7"),
                     use_container_width=True)
        st.bar_chart(df_cmp[["Accuracy", "Macro F1"]])

        st.markdown("""
        **Key takeaways:**
        - The multimodal model achieves **~89% accuracy**, surpassing both unimodal baselines.
        - CNN (face) reaches 82% but is sensitive to lighting and occlusion.
        - BiLSTM (EEG) reaches 65% — EEG signals are noisy but provide ground-truth internal state.
        - Fusion with cross-modal attention recovers information lost by each single modality.
        """)

    # ─────────────────────────────────────────────────────────
    # TAB 3: About
    # ─────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("About this project")
        st.markdown("""
        **Multimodal Negative Emotion Recognition using EEG Signals and Facial Analysis**

        | Component | Detail |
        |-----------|--------|
        | Paper | Group 14 – ML Lab Evaluation Report |
        | EEG dataset | DEAP (Cambridge) / SEED (SJTU) |
        | Face dataset | FER2013 / CK+ |
        | EEG model | 2-layer Bidirectional LSTM + Temporal Attention |
        | Face model | 4-block CNN with BatchNorm |
        | Fusion | Cross-Modal Attention (face ↔ EEG) |
        | Explainability | Grad-CAM · SHAP · Attention Maps |
        | Tuning | Optuna TPE (30 trials) |
        | Deployment | Streamlit + WebRTC webcam |

        **Negative emotions classified:** Anger · Fear · Sadness · Disgust
        """)


if __name__ == "__main__":
    main()
