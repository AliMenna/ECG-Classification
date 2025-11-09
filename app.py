import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Model Definition
# -----------------------------
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

# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_model():
    model = ECGCNN(num_classes=5)
    model.load_state_dict(torch.load("ecg_best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Streamlit
# -----------------------------
st.title("ECG Arrhythmia Classifier")
st.write("Upload an ECG **CSV** file.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'target' in df.columns:
        df = df.drop(columns=['target'])

    st.write("File uploaded!")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    
    index = st.number_input("Choose the row to analyze:", min_value=0, max_value=len(df)-1, value=0)
    signal = df.iloc[index].values.astype(np.float32)

    # ECG Visualization
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal)
    ax.set_title(f"ECG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Prediction
    x = torch.tensor(signal).unsqueeze(0).unsqueeze(0)  # (1,1,180)
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)

    st.subheader("Results")
    st.write(f"**Type of Arrythmia:** {predicted_class}")

    # 
    st.bar_chart(probabilities)
else:
    st.info("Upload a CSV file to start.")
