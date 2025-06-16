import torch
import torch.nn as nn

class DummySentimentModel(nn.Module):
    def __init__(self, input_size=257*341, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def predict(model, spectrogram):
    """
    Run inference on a 2D spectrogram.

    Args:
        model: PyTorch model
        spectrogram: 2D numpy array [freq x time]

    Returns:
        (predicted_class_index, confidence_vector)
    """
    with torch.no_grad():
        x = torch.tensor(spectrogram.flatten()).float().unsqueeze(0)  # [1, N]
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        predicted = int(probs.argmax())
        return predicted, probs
