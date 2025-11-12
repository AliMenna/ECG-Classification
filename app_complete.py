import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# -------------------------------------------------
# 1. CNN MODEL DEFINITION
# -------------------------------------------------
class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 22, 128)  # 180 -> 90 -> 45 -> 22 after pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward propagation
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
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

# Class labels and medical descriptions
CLASS_LABELS = {
    0: "Fusion (F)",
    1: "Normal (N)",
    2: "Unknown (Q)",
    3: "Supraventricular (S)",
    4: "Ventricular (V)"
}

CLASS_DESC = {
    "Normal (N)": "Regular cardiac activity without significant arrhythmia.",
    "Supraventricular (S)": "Premature atrial or junctional beats of supraventricular origin.",
    "Ventricular (V)": "Ventricular premature contractions or ventricular tachycardia.",
    "Fusion (F)": "Fusion between a normal and a premature beat.",
    "Unknown (Q)": "Unclear signal, not confidently assignable to other classes."
}


# -------------------------------------------------
# 3. GRAD-CAM 1D IMPLEMENTATION
# -------------------------------------------------
def grad_cam_1d(model, x, target_class):
    """
    Computes Grad-CAM for a 1D CNN model.
    Args:
        model: trained CNN model
        x: input tensor of shape (1, 1, L)
        target_class: int (true or predicted class)
    Returns:
        cam: Grad-CAM activation map (numpy array)
    """
    model.eval()

    # Hooks to capture activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    # Register hooks on the last conv layer
    handle_f = model.conv3.register_forward_hook(forward_hook)
    handle_b = model.conv3.register_backward_hook(backward_hook)

    # Forward + backward
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    loss = output[0, target_class]
    loss.backward()

    # Get gradients and activations
    grads = gradients["value"]          # (B, C, L)
    acts = activations["value"]         # (B, C, L)

    # Global average pooling of gradients
    weights = grads.mean(dim=2, keepdim=True)   # (B, C, 1)

    # Weighted sum of activations
    cam = (weights * acts).sum(dim=1).clamp(min=0)  # ReLU
    cam = cam.squeeze().cpu().numpy()

    # Normalize for visualization
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    return cam

# -------------------------------------------------
# 4. STREAMLIT DASHBOARD
# -------------------------------------------------
st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")
st.title("💓 ECG Arrhythmia Classifier")
st.markdown("Upload an ECG trace in **CSV** format to automatically classify arrhythmia and visualize model interpretation.")

# -------------------------------------------------
# 📤 FILE UPLOAD AND DATA PREVIEW
# -------------------------------------------------
# ------------------------------------------------------------
# FILE UPLOAD AND DATA PREVIEW (CSV or Image)
# ------------------------------------------------------------
st.subheader("Upload your ECG data")

# Let the user choose between CSV or image input
input_type = st.radio("Select input type:", ["CSV file", "ECG image"])

signal = None  # initialize signal

# ------------------------------------------------------------
# 1️⃣ CSV MODE
# ------------------------------------------------------------
if input_type == "CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load and clean the CSV
            df = pd.read_csv(uploaded_file)
            df = df.drop(columns=['label', 'target', 'class'], errors='ignore')
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                st.error("❌ No numeric data found in this CSV file. Please upload a valid ECG signal file.")
            else:
                st.success(f"✅ File successfully uploaded. Shape: {numeric_df.shape}")

                # Select the row (heartbeat) to analyze
                row_idx = st.slider(
                    "Select the row (heartbeat) to analyze:",
                    0,
                    len(numeric_df) - 1,
                    0
                )

                # Extract the signal
                signal = numeric_df.iloc[row_idx].values.astype(np.float32)
                sig_len = len(signal)
                EXPECTED_LENGTH = 180

                # Adjust signal length
                if sig_len != EXPECTED_LENGTH:
                    if sig_len < EXPECTED_LENGTH:
                        pad_size = EXPECTED_LENGTH - sig_len
                        signal = np.pad(signal, (0, pad_size), mode='constant')
                        st.warning(f"⚠️ Signal too short ({sig_len} samples). Padded to {EXPECTED_LENGTH}.")
                    else:
                        signal = np.interp(
                            np.linspace(0, 1, EXPECTED_LENGTH),
                            np.linspace(0, 1, sig_len),
                            signal
                        )
                        st.warning(f"⚠️ Signal too long ({sig_len} samples). Resampled to {EXPECTED_LENGTH}.")

        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")

# ------------------------------------------------------------
# 2️⃣ IMAGE MODE
# ------------------------------------------------------------
elif input_type == "ECG image":
    uploaded_image = st.file_uploader("Upload an ECG image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded ECG Image", use_column_width=True)

        # Convert to grayscale and process
        img = np.array(image.convert("L"))
        img = cv2.resize(img, (500, 200))
        img = cv2.bitwise_not(img)
        _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # Extract 1D signal from image
        signal = np.mean(thresh, axis=0)
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        EXPECTED_LENGTH = 180
        signal = np.interp(np.linspace(0, 1, EXPECTED_LENGTH),
                           np.linspace(0, 1, len(signal)),
                           signal).astype(np.float32)

        st.success("✅ ECG signal extracted from image.")

# ------------------------------------------------------------
# 3️⃣ MAIN TABS (only if signal is ready)
# ------------------------------------------------------------
if signal is not None:
    tab_signal, tab_result, tab_model = st.tabs(
        ["📈 Signal", "🧠 Classification", "ℹ️ Model Info"]
    )

    # TAB 1 - Display ECG signal
    with tab_signal:
        st.subheader("Selected ECG Signal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode="lines", name="ECG", line=dict(color="black")))
        fig.update_layout(xaxis_title="Samples", yaxis_title="Amplitude", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2 - Classification and Grad-CAM
    with tab_result:
        st.subheader("Classification and Model Explanation")
        if st.button("🔍 Analyze ECG"):
            try:
                # Prepare tensor
                x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = int(np.argmax(probs))

                # Show classification result
                st.markdown(f"**Predicted class:** {CLASS_LABELS[pred_class]}")
                st.info(CLASS_DESC[CLASS_LABELS[pred_class]])

                # Plot Grad-CAM
                st.markdown("### 🔎 Model Attention (Grad-CAM 1D)")
                cam = grad_cam_1d(model, x, pred_class)
                fig_cam = go.Figure()
                fig_cam.add_trace(go.Scatter(y=signal, mode="lines", name="ECG Signal", line=dict(color="black")))
                fig_cam.add_trace(go.Scatter(
                    y=cam * (signal.max() - signal.min()) + signal.min(),
                    mode="lines", name="Grad-CAM Importance", line=dict(color="red")
                ))
                fig_cam.update_layout(
                    title="ECG Signal with Model Attention (Grad-CAM)",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude / Importance",
                    height=350
                )
                st.plotly_chart(fig_cam, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error during classification: {e}")

    # TAB 3 - Model info
    with tab_model:
        st.markdown("""
        ### ℹ️ Model Information
        **Architecture:** 1D CNN (3 conv layers + 2 FC)  
        **Dataset:** MIT-BIH Arrhythmia  
        **Accuracy:** ~83%  
        **AUC average:** 0.95  
        **Frameworks:** PyTorch, Streamlit, Plotly
        """)

