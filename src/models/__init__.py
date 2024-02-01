import torch
from torch import Tensor
import torch.nn as nn
import torchvision

import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod
from utils.metrics import Metrics

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
    '''
    Wraps a Classifier type and implements classify() method
    '''
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


def get_additional_metrics(y_pred, y_target, custom_dict):
    #add_metrics = {
    #         'precision': [],
    #         'recall': [],
    #         'accuracy': [],
    #         'hamming_loss': [],
    #         'jaccard': []
    # }
    metrics = Metrics(y_pred, y_target)

    custom_dict['precision'].append(metrics.precision().cpu())
    custom_dict['recall'].append(metrics.recall().cpu())
    custom_dict['accuracy'].append(metrics.accuracy().cpu())
    custom_dict['hamming_loss'].append(metrics.hamming_loss().cpu())
    custom_dict['jaccard'].append(metrics.jaccard_index().cpu())

    return custom_dict

def get_avg_metrics(custom_dict):
    p = np.mean(custom_dict['precision'])
    r = np.mean(custom_dict['recall'])
    a = np.mean(custom_dict['accuracy'])
    h = np.mean(custom_dict['hamming_loss'])
    j = np.mean(custom_dict['jaccard'])
    return p, r, a, h, j

def train(model, dl_train, optimizer, epochs, device=None):
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    if device:
        model.to(device)

    model.train()

    hist = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'hamming_loss': [],
        'jaccard': []
    } # storing epoch-wise metrics

    for epoch in range(epochs):
        total_loss = 0.0
        add_metrics_dict = {
            'precision': [],
            'recall': [],
            'accuracy': [],
            'hamming_loss': [],
            'jaccard': []
        } # initializes a new metrics dictionary for each epoch

        # Wrap your DataLoader with tqdm for a progress bar
        for batch_idx, (X, y) in enumerate(tqdm(dl_train, desc=f'Epoch {epoch + 1}/{epochs}')):
            if device:
                X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_scores = model(X)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # accumulate metrics across each batch
            add_metrics_dict = get_additional_metrics(model.classify_scores(y_scores), y, add_metrics_dict)

        # Calculate the average loss for the epoch
        average_loss = total_loss / len(dl_train)
        p, r, a, h, j = get_avg_metrics(add_metrics_dict)
        hist['precision'].append(p)
        hist['recall'].append(r)
        hist['accuracy'].append(a)
        hist['hamming'].append(h)
        hist['jaccard'].append(j)

        print(f"\n Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}  -- Precision: {p:.2f} -- Recall: {r:.2f} -- Acc: {a:.2f} -- Hamming: {h:.2f} -- Jaccard: {j:.2f}")

    return model, hist

def test(model, dl_test, device=None):
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    num_correct, loss = 0, 0.0
    if device:
        model.to(device)
    model.eval()

    add_metrics_dict = {
            'precision': [],
            'recall': [],
            'accuracy': [],
            'hamming_loss': [],
            'jaccard': []
        } # initializes a new metrics dictionary

    with torch.no_grad():
        for (x, y) in dl_test:
            if device:
                x, y = x.to(device), y.to(device)
            y_scores = model(x)
            loss += criterion(y_scores, y).item()

            # y_pred = model.classify_scores(y_scores)
            # y_pred = y_pred.cpu().numpy()
            # y = y.cpu().numpy()
            # num_correct += np.sum(y_pred == y)

            add_metrics_dict = get_additional_metrics(model.classify_scores(y_scores), y, add_metrics_dict) # accumulate

    # avg_loss = sum(losses) / num_batches
    #accuracy = 100.0 * np.sum(num_correct) / (len(dl_test.dataset) * 20) # 20: n_classes
    #accuracy = correct / len(dl_test.dataset)
    p, r, a, h, j = get_avg_metrics(add_metrics_dict)
    m = {
        'precision': p,
        'recall': r,
        'accuracy': a,
        'hamming': h,
        'jaccard': j
    }
    return loss, m
