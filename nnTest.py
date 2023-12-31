import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Load the ResNet50 model
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 10), nn.Softmax(dim=1)
)

# Load the saved model weights
model_path = "C:/Users/roach/Desktop/skinner/rete/cose/35 validation.pth"
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def classify_image(model, image):
    with torch.no_grad():
        output = model(image)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_class = torch.max(probabilities, 0)

    return confidence.item(), predicted_class.item()


image_path_to_test = "C:/Users/roach/Desktop/archive (1)/Hair Diseases - Final/test/Folliculitis/folliculitis_0051.jpg"
input_image = preprocess_image(image_path_to_test)

confidence, predicted_class = classify_image(model, input_image)

print(f"Predicted class: {predicted_class}, Confidence: {confidence * 100:.2f}%")
