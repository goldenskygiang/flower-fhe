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

class BinaryClassifier(Classifier):
    '''
    Used for BINARY classification (also works for multi-label)
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


def get_model(ds: str='pascal', num_classes: int=20, threshold: float=0.5, model_choice: str='mobilenet', dropout: float=0.4):
    '''
    Args:
        @ds           -- The dataset / task for the model.
                         If 'pascal' => Multi-label classification, num_classes=20
                         if 'cifar'  => Multi-class classification, num_classes=[10, 100]
        @num_classes  -- Number of output classes. 20 for PascalVOC multilabel, [10, 100] for Cifar multiclass
        @threshold    -- Prediction threshold for Binary Classification (or multi-label)
        @model_choice -- The backbone CNN model. Either 'mobilenet' or 'resnet' atm
        @dropout      -- Dropout probability for the classification head's dropout layer

    Returns:
        @model        -- torch.nn.Module() model
    '''
    model = None
    in_features = 0

    if ds == 'pascal':
        assert num_classes==20, "Pascal VOC Dataset requires num_classes=20"
    if ds == 'cifar':
        assert num_classes==10 or num_classes==100, "Cifar Dataset requires num_classes either 10 or 100"

    if model_choice == 'mobilenet':
        model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features

    elif model_choice == 'resnet':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        # model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Sequential(nn.Dropout(0.2),
        #                         nn.Linear(model.fc.in_features, 64),
        #                         nn.ReLU(),
        #                         nn.Linear(64, num_classes))
        in_features = model.fc.in_features

    elif model_choice == 'mnasnet':
        model = torchvision.models.mnasnet0_75(weights='IMAGENET1K_V1')
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.mnasnet0_75.html#torchvision.models.MNASNet0_75_Weights
        in_features = model.classifier[1].in_features

    else:
        raise ValueError('Model type not supported.')

    ## Freeze all pretrained layers
    for p in model.parameters():
        p.requires_grad = False

    ## unfreeze last conv layers for transfer learning
    if model_choice == 'mnasnet':
        unfreeze_layers=["layers.14", "layers.15", "classifier"]
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):  # Check if name matches any layer in the list
                param.requires_grad = True
                print(f'Unfreezing {name}')

    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
        #nn.Sigmoid()  # Sigmoid is in either the loss fn or the wrapper class
    )

    ## Replaces classification head
    if model_choice == 'mobilenet' or model_choice == 'mnasnet':
        model.classifier = classifier
    else: # resnet atm
        model.fc = classifier

    ## Implements model.classify() method
    if ds == 'pascal':
        model = BinaryClassifier(model, threshold=threshold)
    else: # 'cifar
        model = MultiClassClassifier(model)

    return model

def generate_model_fn(**kwargs):
    def model_fn():
        return get_model(**kwargs)
    return model_fn

# Standard Train / Test loop for MultiLabel Task
def train(ds, model, dl_train, optimizer, epochs, device=None, proximal_mu: float=0):
    if ds == 'pascal':
        criterion = nn.BCEWithLogitsLoss(reduction='sum') # For multilabel classification
    else: # 'cifar'
        criterion = nn.CrossEntropyLoss() # For multiclass classification

    if device:
        model.to(device)

    global_params = [val.clone() for val in model.parameters()]

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_correct = 0

        for batch_idx, (X, y) in enumerate(tqdm(dl_train, desc=f'Epoch {epoch + 1}/{epochs}')):
            if device:
                X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            proximal_term = 0.0
            for local_w, global_w in zip(model.parameters(), global_params):
                proximal_term += torch.square((local_w - global_w).norm(2))

            y_scores = model(X) # forward
            loss = criterion(y_scores, y) + (proximal_mu / 2) * proximal_term
            loss.backward() # backward
            optimizer.step()
            total_loss += loss.item()

            # Acc
            y_pred = model.classify_scores(y_scores)
            if ds == 'pascal':
                y_pred = y_pred.cpu().numpy()
                y = y.cpu().numpy()
                num_correct += np.sum(y_pred == y)
            else:
                num_correct += torch.sum(y_pred == y).item()

        # Average loss for the epoch
        average_loss = total_loss / len(dl_train)
        # Average accuracy
        if ds == 'pascal':
            accuracy = np.sum(num_correct) / (len(dl_train) * 20) # 20: n_classes
        else:
            accuracy = num_correct / len(dl_train.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Acc: {accuracy:.4f}")

    return model


def test(ds, model, dl_test, device=None):
    if ds == 'pascal':
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    else: # 'cifar'
        criterion = nn.CrossEntropyLoss()

    num_correct, loss = 0, 0.0

    if device:
        model.to(device)
    model.eval()

    with torch.no_grad():
        for (X, y) in dl_test:
            if device:
                X, y = X.to(device), y.to(device)

            y_scores = model(X)
            loss += criterion(y_scores, y).item()

            y_pred = model.classify_scores(y_scores)
            if ds == 'pascal':
                y_pred = y_pred.cpu().numpy()
                y = y.cpu().numpy()
                num_correct += np.sum(y_pred == y)
            else:
                num_correct += torch.sum(y_pred == y).item()

    # avg_loss = sum(losses) / num_batches
    if ds == 'pascal':
        accuracy = np.sum(num_correct) / (len(dl_test.dataset) * 20) # 20: n_classes
    else: # 'cifar'
        accuracy = num_correct / len(dl_test.dataset)
    #accuracy = correct / len(dl_test.dataset)
    return loss, accuracy


# Standard Train / Test loop for CIFAR
# def train_cifar(model, dl_train, optimizer, epochs, device=None):
#     criterion = nn.CrossEntropyLoss()

#     if device:
#         model = model.to(device)
#     model.train()

#     for epoch in range(epochs):
#         total_loss = 0.0
#         num_correct = 0

#         for batch_idx, (X, y) in enumerate(tqdm(dl_train, desc=f'Epoch {epoch + 1}/{epochs}')):
#             if device:
#                 X, y = X.to(device), y.to(device)

#             optimizer.zero_grad()

#             y_scores = model(X) # forward
#             loss = criterion(y_scores, y)
#             loss.backward() # backward
#             optimizer.step()
#             total_loss += loss.item()

#             y_pred = model.classify_scores(y_scores)
#             num_correct += torch.sum(y_pred == y).item()
#             #print(f'{num_correct} -')
#             #print(f'{y_pred==y} -')

#         # Calculate the average loss for the epoch
#         average_loss = total_loss / len(dl_train)
#         acc = num_correct / len(dl_train)

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Acc: {acc:.4f}")

#     return model

# def test_cifar(model, dl_test, device=None):
#     criterion = nn.CrossEntropyLoss()

#     num_correct, loss = 0, 0.0

#     if device:
#         model.to(device)
#     model.eval()

#     with torch.no_grad():
#         for (X, y) in dl_test:
#             if device:
#                 X, y = X.to(device), y.to(device)

#             y_scores = model(X) # forward
#             loss += criterion(y_scores, y).item()

#             y_pred = model.classify_scores(y_scores)
#             num_correct += torch.sum(y_pred == y).item()

#     accuracy = num_correct / len(dl_test)
#     return loss, accuracy

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