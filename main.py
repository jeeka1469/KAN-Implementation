import streamlit as st
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mlp_model import MLP  # Import the MLP model
from kan_model import KANModelWithGridExtension  # Import the KAN model

# Load the saved MLP model
def load_mlp_model():
    model = MLP(input_dim=28*28, hidden_dim=128, output_dim=10)
    model.load_state_dict(torch.load("mlp_model.pt"))
    model.eval()
    return model

# Load the saved KAN model
def load_kan_model():
    model = KANModelWithGridExtension(input_dim=28*28, hidden_dim=128, output_dim=10, num_splines=20)
    model.load_state_dict(torch.load("kan_model.pt"))
    model.eval()
    return model

# Function to evaluate a model
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    return accuracy

# Load test data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load models
mlp_model = load_mlp_model()
kan_model = load_kan_model()

# Streamlit UI
st.title("Model Comparison: MLP vs KAN with Grid Extension")
st.write("Choose a model to evaluate:")

# Model selection
model_choice = st.selectbox("Choose Model", ["MLP", "KAN with Grid Extension"])

# Evaluate the selected model
if model_choice == "MLP":
    st.write("Evaluating MLP model...")
    accuracy = evaluate_model(mlp_model, test_loader)
    st.write(f"MLP Test Accuracy: {accuracy:.2f}%")
elif model_choice == "KAN with Grid Extension":
    st.write("Evaluating KAN model...")
    accuracy = evaluate_model(kan_model, test_loader)
    st.write(f"KAN Test Accuracy: {accuracy:.2f}%")