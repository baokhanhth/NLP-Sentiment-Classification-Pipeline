import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification

def build_model(model_name="vinai/phobert-base", num_labels=3, device=None):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to(device)

    
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        cm = confusion_matrix(labels, preds, labels=[0,1,2])

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        eps = 1e-8  
        accuracy = np.sum(TP) / np.sum(cm)

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        specificity = TN / (TN + FP + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        report = classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "accuracy": accuracy,

            "macro_precision": np.mean(precision),
            "macro_recall": np.mean(recall),
            "macro_specificity": np.mean(specificity),
            "macro_f1": np.mean(f1),

            "precision_class0": precision[0],
            "precision_class1": precision[1],
            "precision_class2": precision[2],

            "recall_class0": recall[0],
            "recall_class1": recall[1],
            "recall_class2": recall[2],

            "specificity_class0": specificity[0],
            "specificity_class1": specificity[1],
            "specificity_class2": specificity[2],

            "f1_class0": f1[0],
            "f1_class1": f1[1],
            "f1_class2": f1[2],

            "TP_class0": TP[0], "TN_class0": TN[0],
            "FP_class0": FP[0], "FN_class0": FN[0],

            "TP_class1": TP[1], "TN_class1": TN[1],
            "FP_class1": FP[1], "FN_class1": FN[1],

            "TP_class2": TP[2], "TN_class2": TN[2],
            "FP_class2": FP[2], "FN_class2": FN[2],
        }

        return metrics

    return model, compute_metrics
