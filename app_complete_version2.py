import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# -------------------------------------------------
# 1. CNN MODEL DEFINITION
# -------------------------------------------------
class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(128 * 22, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -------------------------------------------------
# 2. LOAD TRAINED MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = ECGCNN(num_classes=5)
    model.load_state_dict(torch.load("ecg_best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()


# -------------------------------------------------
# 3. CLASS LABELS
# -------------------------------------------------
CLASS_LABELS = {
    0: "Fusion (F)",
    1: "Normal (N)",
    2: "Unknown (Q)",
    3: "Supraventricular (S)",
    4: "Ventricular (V)"
}

CLASS_DESC = {
    "Normal (N)": "Regular cardiac activity without significant arrhythmia.",
    "Supraventricular (S)": "Premature atrial or junctional beats.",
    "Ventricular (V)": "Ventricular premature contractions or tachycardia.",
    "Fusion (F)": "Fusion between a normal and a premature beat.",
    "Unknown (Q)": "Unclear signal, hard to classify reliably."
}


# -------------------------------------------------
# 4. R-PEAK DETECTION + CLINICAL FEATURES
# -------------------------------------------------
def detect_r_peaks(signal, fs=180):
  
    signal = np.array(signal)
    peaks = []

    threshold = np.mean(signal) + 0.5 * np.std(signal)

    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)


    min_distance = int(0.2 * fs)
    clean_peaks = []
    last_peak = -min_distance

    for p in peaks:
        if p - last_peak >= min_distance:
            clean_peaks.append(p)
            last_peak = p

    return np.array(clean_peaks)



# -------------------------------------------------
# 5. STREAMLIT DASHBOARD
# -------------------------------------------------
st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")
st.title("💓 ECG Arrhythmia Classifier")
st.write("Upload an ECG **CSV** file to classify arrhythmia and analyze the heartbeat.")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['label', 'target', 'class'], errors='ignore')
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            st.error("❌ No valid numeric data found in this CSV.")
        else:
            st.success(f"✅ File loaded. Shape: {numeric_df.shape}")

            row_idx = st.slider("Select heartbeat row:", 0, len(numeric_df) - 1, 0)

            signal = numeric_df.iloc[row_idx].values.astype(np.float32)
            sig_len = len(signal)
            EXPECTED_LENGTH = 180

            if sig_len != EXPECTED_LENGTH:
                if sig_len < EXPECTED_LENGTH:
                    pad = EXPECTED_LENGTH - sig_len
                    signal = np.pad(signal, (0, pad))
                    st.warning(f"⚠️ Signal too short ({sig_len}). Padded to {EXPECTED_LENGTH}.")
                else:
                    signal = np.interp(np.linspace(0, 1, EXPECTED_LENGTH),
                                       np.linspace(0, 1, sig_len),
                                       signal)
                    st.warning(f"⚠️ Signal too long ({sig_len}). Resampled to {EXPECTED_LENGTH}.")

            tab_signal, tab_result, tab_model = st.tabs(
                ["📈 Signal", "🧪 Classification", "ℹ️ Model Info"]
            )

            # TAB 1 — ECG + R-peaks
            with tab_signal:
                st.subheader("ECG Signal")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=signal, mode="lines",
                                         line=dict(color="black"), name="ECG"))
                fig.update_layout(height=300, xaxis_title="Samples", yaxis_title="Amplitude")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🔎 R-Peak Detection")
                peaks = detect_r_peaks(signal)

                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(y=signal, mode="lines",
                                           line=dict(color="black"), name="ECG"))

                if len(peaks) > 0:
                    fig_r.add_trace(go.Scatter(
                        x=peaks, y=signal[peaks], mode="markers",
                        marker=dict(color="red", size=8),
                        name="R Peaks"
                    ))

                fig_r.update_layout(title="ECG with R-Peak Detection",
                                    height=300, xaxis_title="Samples")
                st.plotly_chart(fig_r, use_container_width=True)

                st.subheader("🩺 ECG Feature Analysis")
                features = ecg_features(signal, peaks)
                st.json(features)

            # TAB 2 — Classification
            with tab_result:
                st.subheader("Classification Result")

                x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = int(np.argmax(probs))

                st.markdown(f"### Predicted class: **{CLASS_LABELS[pred_class]}**")
                st.info(CLASS_DESC[CLASS_LABELS[pred_class]])

                fig_bar = px.bar(
                    x=[CLASS_LABELS[i] for i in range(5)],
                    y=probs,
                    labels={"x": "Class", "y": "Probability"},
                    title="Prediction Probabilities"
                )
                fig_bar.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_bar, use_container_width=True)

            # TAB 3 — Model Info
            with tab_model:
                st.markdown("""
                ### 🧩 Model Information
                **Architecture:** 1D CNN (3 conv + 2 FC layers)  
                **Dataset:** MIT-BIH Arrhythmia  
                **Reported Accuracy:** ~83%  
                **AUC:** ~0.95  
                **Libraries:** PyTorch, Streamlit, Plotly  
                """)

    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
