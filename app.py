import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Apply custom background color
st.markdown(
    """
    <style>
    body {
        background-color:rgb(219, 70, 192);
        color: black;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgb(187, 187, 187);
        color: black;
    }

    [data-testid="stAppViewContainer"] {
        background-color:rgb(54, 55, 57);
    }
    
    [data-testid="stHeader"] {
        background-color:rgb(54, 55, 57);
    
    </style>
    """,
    unsafe_allow_html=True
)
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

tomato_model, tomato_device = load_model("Tomato1-EfficientNetB3_model.pth", num_classes=10)

# Load Lettuce Model (8 classes)
lettuce_model, lettuce_device = load_model("Lettuce-EfficientNetB3_model.pth", num_classes=8)
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
tomato_class_names = [
    "Bacterial_spot", "Early_blight", "Late_blight",
    "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "healthy"
]

lettuce_class_names = [
    "Bacterial", "Downy", "Fungal", "Healthy", 
    "Powdery", "Septoria", "Viral", "Wilt"
]

with st.sidebar:
    st.header("🌱 Plant Disease Classifier")
    st.write(
        """
        Welcome to the Plant Disease Classification Web App! 🌿
        
        This tool allows you to upload images of tomato and lettuce leaves to detect potential diseases using advanced deep learning models.
        
        Simply upload an image and click 'Predict' to see the diagnosis.
        """
    )


tabs = st.tabs(["🍅 Tomato Disease Classification", "🥬 Lettuce Disease Classification"])
# ---------------------------
# 4. Streamlit UI
# ---------------------------
with tabs[0]:
    st.title("🍅 Tomato Disease Classification Demo")
    st.write("Upload an image of a tomato leaf, and the model will predict its health condition.")

    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="tomato_uploader")

    if uploaded_files:
        if st.button("Predict", key="predict_tomato"):
            st.write("### Predictions")
            images_list = []
            predicted_labels = []

            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                input_tensor = test_transform(image).unsqueeze(0).to(tomato_device)

                with torch.no_grad():
                    outputs = tomato_model(input_tensor)
                    _, pred = torch.max(outputs, 1)  # Use raw logits directly

                predicted_label = tomato_class_names[pred.item()]
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
with tabs[1]:
    st.title("🥬 Lettuce Disease Classification Demo")
    st.write("Upload an image of a lettuce leaf, and the model will predict its health condition.")

    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="lettuce_uploader")

    if uploaded_files:
        if st.button("Predict", key="predict_lettuce"):
            st.write("### Predictions")
            images_list = []
            predicted_labels = []

            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                input_tensor = test_transform(image).unsqueeze(0).to(lettuce_device)

                with torch.no_grad():
                    outputs = lettuce_model(input_tensor)
                    _, pred = torch.max(outputs, 1)  # Use raw logits directly

                predicted_label = lettuce_class_names[pred.item()]
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