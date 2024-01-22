import torch
from torch import Tensor
import torch.nn as nn
import torchvision

from abc import ABC, abstractmethod

class Classifier(nn.Module, ABC):
    '''
    Wraps a model which produces raw class scores, and provides methods to compute
        class label and probabilities
    '''
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


    def forward(self, X: Tensor) -> Tensor:
        z: Tensor = None
        z = self.model(X)
        assert z.shape[0] == X.shape[0]
        return z

    def predict_proba(self, X: Tensor) -> Tensor:
        z = self.model(X)
        return self.predict_proba_scores(z)

    def predict_proba_scores(self, z: Tensor) -> Tensor:
        return nn.functional.sigmoid(z)

    def classify(self, X: Tensor) -> Tensor:
        y_proba = self.predict_proba(X)

        return self._classify(y_proba)

    def classify_scores(self, z: Tensor) -> Tensor:
        y_proba = self.predict_proba_scores(z)

        return self._classify(y_proba)

    @abstractmethod
    def _classify(self, y_proba: Tensor) -> Tensor:
        pass

class MultiLabelClassifier(Classifier):
    def __init__(self, model, threshold: float = 0.5):
        super().__init__(model)
        self.threshold = threshold

    def _classify(self, y_proba: Tensor):
        return (y_proba > self.threshold).type(torch.float32)

def get_model(num_classes=20, threshold=0.5):
    mobilenet = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

    for p in mobilenet.parameters():
        p.requires_grad = False

    in_features = mobilenet.classifier[1].in_features
    #num_classes = 20  # Number of classes for multi-label classification

    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        # nn.Conv2d(in_features, 256, kernel_size=1),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        #nn.Sigmoid()  # Sigmoid is in either the loss fn or the wrapper class
    )

    mobilenet.classifier = classifier

    mobilenet = MultiLabelClassifier(mobilenet, threshold=threshold)

    return mobilenet