import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class HairDiseaseClassifier:
    def __init__(self, model_path, class_labels, temperature=1.0):
        self.class_labels = class_labels
        self.temperature = temperature
        self.model = self._load_model(model_path, class_labels)

    def _load_model(self, model_path, class_labels):
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, len(class_labels)), nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def preprocess_image(self, image_path):
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

    def classify_image(self, image):
        with torch.no_grad():
            # Pass the input through temperature scaling
            scaled_output = self._temperature_scale(self.model(image))

        probabilities = torch.nn.functional.softmax(scaled_output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

        return confidence.item(), predicted_class.item()

    def get_class_label(self, class_index):
        return self.class_labels[class_index]

    def _temperature_scale(self, logits):
        scaled_logits = logits / self.temperature
        return scaled_logits

def main():
    model_path = "./trained_model_weights/epoch_17.pth"
    class_labels = [
        "Alopecia Areata", "Psoriasis", "Folliculitis", "Seborrheic Dermatitis",
        "Lichen Planus", "Male Pattern Baldness", "Contact Dermatitis", "Head Lice",
        "Telogen Effluvium", "Tinea Capitis"
    ]

    temperature = 0.1  

    classifier = HairDiseaseClassifier(model_path, class_labels, temperature)

    image_path_to_test = "./trainingDataset/Hair Diseases - Final/test/Head Lice/head_lice_0051.jpg"
    input_image = classifier.preprocess_image(image_path_to_test)

    confidence, predicted_class = classifier.classify_image(input_image)
    predicted_label = classifier.get_class_label(predicted_class)

    print(f"Predicted class: {predicted_label}, Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
