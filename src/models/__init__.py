import torch
from torch import Tensor
import torch.nn as nn
import torchvision

import numpy as np
from tqdm import tqdm

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
    '''
    Used for BINARY classification
    Label y_i = [0/1] * n_classes
    '''
    def __init__(self, model, threshold: float = 0.5):
        super().__init__(model)
        self.threshold = threshold

    def _classify(self, y_proba: Tensor):
        return (y_proba > self.threshold).type(torch.float32)

class MultiClassClassifier(Classifier):
    '''
    Used for MULTICLASS classification
    Label y_i = [0 / 1 / ... / n_classes - 1]
    '''
    def __init__(self, model):
        super().__init__(model)

    def _classify(self, y_proba: Tensor):
        return torch.argmax(y_proba, dim=1)


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


def get_model_cifar(num_classes: int=10, choice: str='mobilenet'):
    model = None
    if choice == 'resnet':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        in_features = model.fc.in_features
        # model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Sequential(nn.Dropout(0.2),
        #                         nn.Linear(model.fc.in_features, 64),
        #                         nn.ReLU(),
        #                         nn.Linear(64, num_classes))
    else:
        model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features

    # Freeze all layers except the classifier
    for p in model.parameters():
        p.requires_grad = False

    # classifier head
    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    model.classifier = classifier # replace with new head
    model = MultiClassClassifier(model)

    return model

# Standard Train / Test loop for MultiLabel Task
def train(model, dl_train, optimizer, epochs, device=None, proximal_mu: float=0):
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    if device:
        model.to(device)

    global_params = [val.clone() for val in model.parameters()]

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        # Wrap your DataLoader with tqdm for a progress bar
        for batch_idx, (X, y) in enumerate(tqdm(dl_train, desc=f'Epoch {epoch + 1}/{epochs}')):
            if device:
                X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            proximal_term = 0.0
            for local_w, global_w in zip(model.parameters(), global_params):
                proximal_term += torch.square((local_w - global_w).norm(2))

            y_scores = model(X)
            loss = criterion(y_scores, y) + (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate the average loss for the epoch
        average_loss = total_loss / len(dl_train)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    return model

def test(model, dl_test, device=None):
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    num_correct, loss = 0, 0.0
    if device:
        model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dl_test:
            if device:
                x, y = x.to(device), y.to(device)
            y_scores = model(x)
            loss += criterion(y_scores, y).item()

            y_pred = model.classify_scores(y_scores)
            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()
            num_correct += np.sum(y_pred == y)

    # avg_loss = sum(losses) / num_batches
    accuracy = 100.0 * np.sum(num_correct) / (len(dl_test.dataset) * 20) # 20: n_classes
    #accuracy = correct / len(dl_test.dataset)
    return loss, accuracy

# Standard Train / Test loop for CIFAR
def train_cifar(model, dl_train, optimizer, epochs, device=None):
    criterion = nn.CrossEntropyLoss()

    if device:
        model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_correct = 0

        for batch_idx, (X, y) in enumerate(tqdm(dl_train, desc=f'Epoch {epoch + 1}/{epochs}')):
            if device:
                X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            y_scores = model(X) # forward
            loss = criterion(y_scores, y)
            loss.backward() # backward
            optimizer.step()
            total_loss += loss.item()

            y_pred = model.classify_scores(y_scores)
            num_correct += torch.sum(y_pred == y).item()
            #print(f'{num_correct} -')
            #print(f'{y_pred==y} -')

        # Calculate the average loss for the epoch
        average_loss = total_loss / len(dl_train)
        acc = num_correct / len(dl_train)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Acc: {acc:.4f}")

    return model

def test_cifar(model, dl_test, device=None):
    criterion = nn.CrossEntropyLoss()

    num_correct, loss = 0, 0.0

    if device:
        model.to(device)
    model.eval()

    with torch.no_grad():
        for (X, y) in dl_test:
            if device:
                X, y = X.to(device), y.to(device)

            y_scores = model(X) # forward
            loss += criterion(y_scores, y).item()

            y_pred = model.classify_scores(y_scores)
            num_correct += torch.sum(y_pred == y).item()

    accuracy = num_correct / len(dl_test)
    return loss, accuracy

# def run_centralised_cifar(epochs: int, lr: float, momentum: float=0.9):
#     model = models.get_model_cifar()

#     # define optimizer
#     optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#     # get dataset, construct ds and dl
#     train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
#     test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

#     # train loop
#     trained_model = train_cifar(model, train_dl, optim, epochs)

#     # evaluate after training
#     loss, accuracy = test_cifar(model, test_dl)
#     print(f"{loss=}")
#     print(f"{accuracy=}")

#     return trained_model

# trained_model = run_centralised_cifar(epochs=30, lr=0.01)