from torchmetrics import Metric
import torch

from sklearn.metrics import accuracy_score, f1_score  # -> 확인용


# [TODO] Implement this!
# class MyF1Score(Metric):
#     pass
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        pred_labels = torch.argmax(preds, dim=1)

        # Compute true positives, false positives, and false negatives for each class
        for class_idx in range(preds.shape[1]):  # Calculate per-class F1 score in a one-vs-rest manner
            class_preds = (pred_labels == class_idx).float()
            class_targets = (target == class_idx).float()

            true_positives = torch.sum(class_preds * class_targets).long()
            false_positives = torch.sum((class_preds == 1) & (class_targets == 0)).long()
            false_negatives = torch.sum((class_preds == 0) & (class_targets == 1)).long()

            self.true_positives += true_positives
            self.false_positives += false_positives
            self.false_negatives += false_negatives

    def compute(self):
        precision = self.true_positives.float() / (self.true_positives.float() + self.false_positives.float() + 1e-9)
        recall = self.true_positives.float() / (self.true_positives.float() + self.false_negatives.float() + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return f1  # torch.mean(f1)  # print(f1.shape)
    

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        # print(preds.shape, target.shape)
        pred_labels = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert pred_labels.shape == target.shape
        # assert preds.shape == target.shape

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(pred_labels == target)
        # print('check correct of myacc:', correct, preds.shape)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

        # 확인용
        # accuracy = accuracy_score(target.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())  # -> ok
        # print("Acc:", accuracy)
        # f1 = f1_score(target.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), average='weighted')
        # print("F1 Score:", f1)

    def compute(self):
        return self.correct.float() / self.total.float()
