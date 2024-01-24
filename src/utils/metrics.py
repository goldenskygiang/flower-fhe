import torch

class Metrics:
    def __init__(self, prediction, target):
        self.batch_size = prediction.shape[0]
        self.num_labels = prediction.shape[1]
        self.prediction = prediction
        self.target = target
        self.tp, self.tn, self.fp, self.fn = self._get_tf()
        self.tpl, self.tnl, self.fpl, self.fnl = self._get_tf_label()

    def _get_tf(self):
        true_positives = torch.sum((self.prediction == 1) & (self.target == 1)).float()
        true_negatives = torch.sum((self.prediction == 0) & (self.target == 0)).float()
        false_positives = torch.sum((self.prediction == 1) & (self.target == 0)).float()
        false_negatives = torch.sum((self.prediction == 0) & (self.target == 1)).float()
        return true_positives, true_negatives, false_positives, false_negatives

    def _get_tf_label(self):
        '''
        Label-Wise metrics

        returns: 4 tensors, each of shape (batch_size, num_labels)
        '''
        true_positives_label = torch.sum((self.prediction == 1) & (self.target == 1), axis=0).float()
        true_negatives_label = torch.sum((self.prediction == 0) & (self.target == 0), axis=0).float()
        false_positives_label = torch.sum((self.prediction == 1) & (self.target == 0), axis=0).float()
        false_negatives_label = torch.sum((self.prediction == 0) & (self.target == 1), axis=0).float()
        return true_positives_label, true_negatives_label, false_positives_label, false_negatives_label

    def accuracy(self):
        num_correct = torch.sum((self.prediction == self.target)).float()
        total_entries = self.batch_size * self.num_labels
        acc = num_correct / total_entries
        return acc

    def precision(self):
        precision = self.tp / (self.tp + self.fp + 1e-10)
        return precision

    def recall(self):
        recall = self.tp / (self.tp + self.fn + 1e-10)
        return recall

    def precision_label(self):
        precision_label = self.tpl / (self.tpl + self.fpl + 1e-10) # element-wise
        return precision_label

    def recall_label(self):
        recall_label = self.tpl / (self.tpl + self.fnl + 1e-10) # element-wise
        return recall_label

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

    def jaccard_index(self):
        intersection = self.tp
        union = torch.sum((self.prediction.int() | self.target.int()) == 1).float()
        j_index = intersection / (union + 1e-10)
        return j_index

    def hamming_loss(self):
        hamming_loss = torch.mean((self.prediction != self.target).float())
        return hamming_loss

    def __repr__(self):
        return (f"""batch_size={self.batch_size} -- Precision={self.precision()} -- Recall={self.recall()} -- F1={self.f1()}
                accuracy={self.accuracy()} -- jaccard_index={self.jaccard_index()} -- hamming_loss={self.hamming_loss()}""")

