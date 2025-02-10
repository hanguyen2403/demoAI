import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# 1. Load the Pretrained Model
# ---------------------------
@st.cache_resource
def load_model(checkpoint_path, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Match the architecture with the training model
    model = models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, num_classes)
    )
    
    # Load model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# Adjust the checkpoint path if necessary.
checkpoint_path = "Tomato-EfficientNetB3_model.pth"
model, device = load_model(checkpoint_path, num_classes=10)
st.write("Model loaded successfully!")

# ---------------------------
# 2. Define the Transformation
# ---------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------
# 3. Define Class Names Mapping
# ---------------------------
class_names = [
    "Bacterial_spot", "Early_blight", "Late_blight",
    "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "healthy"
]

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("Smart Vertical Farming: Tomato Disease Detection")
st.write("Select one or more image files using the file uploader below:")

uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Predict"):
        st.write("### Predictions")
        images_list = []
        predicted_labels = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            input_tensor = test_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)  # Use raw logits directly

            predicted_label = class_names[pred.item()]
            images_list.append(image)
            predicted_labels.append(predicted_label)

        # Display Images
        num_cols = 5
        num_images = len(images_list)
        for i in range(0, num_images, num_cols):
            cols = st.columns(num_cols)
            for j, image in enumerate(images_list[i:i + num_cols]):
                caption = f"Predicted: {predicted_labels[i + j]}"
                cols[j].image(image, caption=caption, use_container_width=True)
