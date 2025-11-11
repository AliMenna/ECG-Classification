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
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
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

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # -------------------------------------------------
    # Read data
    # -------------------------------------------------
    df = pd.read_csv(uploaded_file, header=None)
    if 'target' in df.columns:
        df = df.drop(columns=['target'])

    st.success(f"File successfully uploaded. Shape: {df.shape}")
    with st.expander("Preview data"):
        st.dataframe(df.head(), use_container_width=True)

    # Select which row (beat) to analyze
    row_idx = st.slider("Select the row (heartbeat) to analyze:", 0, len(df) - 1, 0)
    signal = df.iloc[row_idx].values.astype(np.float32)
    sig_len = len(signal)

    # -------------------------------------------------
    # MAIN TABS
    # -------------------------------------------------
    tab_signal, tab_result, tab_compare, tab_model = st.tabs(
        ["📈 Signal", "📊 Classification", "🩺 Typical patterns", "🧪 Model Info"]
    )

    # -------------------------------------------------
    # TAB 1 - Display signal
    # -------------------------------------------------
    with tab_signal:
        st.subheader("Selected ECG signal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode="lines", name="ECG"))
        fig.update_layout(xaxis_title="Samples", yaxis_title="Amplitude", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Make prediction
    # -------------------------------------------------
    x = torch.tensor(signal).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))

    # -------------------------------------------------
    # TAB 2 - Classification result
    # -------------------------------------------------
    with tab_result:
        st.subheader("Classification results")

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            bar_fig = px.bar(
                x=[CLASS_LABELS[i] for i in range(len(probs))],
                y=probs,
                title="Class probabilities",
                labels={"x": "Class", "y": "Probability"},
            )
            bar_fig.update_yaxes(range=[0, 1])
            st.plotly_chart(bar_fig, use_container_width=True)

        with col_right:
            st.markdown(f"**Predicted class:** `{CLASS_LABELS[pred_class]}`")
            st.info(CLASS_DESC[CLASS_LABELS[pred_class]])

            # Download CSV report
            report_df = pd.DataFrame({
                "row": [row_idx],
                "predicted_class": [CLASS_LABELS[pred_class]],
                "probabilities": [probs.tolist()]
            })
            st.download_button(
                "📥 Download report (CSV)",
                report_df.to_csv(index=False),
                file_name=f"ecg_report_row{row_idx}.csv",
                mime="text/csv"
            )

        # ---------------- GRAD-CAM ----------------
        st.markdown("### 🔎 Model Explanation (Grad-CAM 1D)")
        cam = grad_cam_1d(model, x, pred_class)


        # Combined ECG + Grad-CAM visualization
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(y=signal, mode="lines", name="ECG", line=dict(color="black")))
        fig_cam.add_trace(go.Scatter(
            y=cam * (signal.max() - signal.min()) + signal.min(),
            mode="lines",
            name="Importance (Grad-CAM)",
            line=dict(color="red")
        ))
        fig_cam.update_layout(
            title="ECG signal with model attention overlay (Grad-CAM)",
            xaxis_title="Samples",
            yaxis_title="Amplitude / Importance",
            height=300
        )
        st.plotly_chart(fig_cam, use_container_width=True)

    # -------------------------------------------------
    # TAB 3 - Typical waveforms
    # -------------------------------------------------
    with tab_compare:
        st.subheader("Compare with typical class waveforms")

        typical_shapes = {}
        # Load average waveform examples if available
        for label in CLASS_LABELS.values():
            file_name = f"avg_{label.split()[0]}.npy"
            if os.path.exists(file_name):
                typical_shapes[label] = np.load(file_name)

        if typical_shapes:
            selected_classes = st.multiselect(
                "Select classes to compare:",
                list(typical_shapes.keys()),
                default=[CLASS_LABELS[pred_class]]
            )
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(y=signal, mode="lines", name="Uploaded ECG", line=dict(color="black")))
            for c in selected_classes:
                fig_cmp.add_trace(go.Scatter(y=typical_shapes[c], mode="lines", name=f"Typical {c}"))
            fig_cmp.update_layout(
                title="Comparison with typical ECG patterns",
                xaxis_title="Samples",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.warning("No typical waveform files found (e.g. avg_N.npy).")

    # -------------------------------------------------
    # TAB 4 - Model & performance info
    # -------------------------------------------------
    with tab_model:
        st.markdown("""
        ### Technical Information  
        - **Architecture:** 1D Convolutional Neural Network (3 conv + 2 FC layers)  
        - **Training dataset:** MIT-BIH Arrhythmia  
        - **Reported accuracy:** ~83%  
        - **AUC average:** 0.95  
        - **Libraries:** PyTorch, Streamlit, Plotly  
        """)
