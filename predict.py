import torch
import config
import numpy as np
from data import loadTestData
from model import loadModel
from customClassifier import ClocktestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
#from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


from sklearn.metrics import roc_auc_score
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict



def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df



use_gpu = True
batch_size = config.batch_size

# Load the model
trained_model = loadModel(config.model_type)
# create a complete CNN
trained_model.classifier = ClocktestClassifier()

# Load the network weights with the Lowest Validation Loss
checkPoint = torch.load(config.output_best_checkpoit)
trained_model.load_state_dict(checkPoint['state_dict'], strict=False)
print("The Model was Loaded Successfully")
print("")

# move tensors to GPU if CUDA is available
if use_gpu:
    trained_model.cuda()

print("Load test data ...")
test_loader = loadTestData(batch_size)

print("Testing the Trained Network is in Progress ...")
print("")
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(config.n_class))
class_total = list(0. for i in range(config.n_class))
confusion_matrix = torch.zeros(config.n_class, config.n_class)
target_all_np = []
pred_all_np =[]
pred_all_offdiagonal_np = []

trained_model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = trained_model(data)

    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = correct_tensor.numpy() if not use_gpu else correct_tensor.cpu().numpy()
    # calculate test accuracy for each object class
    for i in range(len(correct.data)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

    # fill confusion matrix
    for t, p in zip(target.data.view(-1), pred.data.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # concatnate for auc calculation
    target_np = target.numpy() if not use_gpu else target.cpu().numpy()
    pred_np = pred.numpy() if not use_gpu else pred.cpu().numpy()
    target_all_np.extend(target_np)
    pred_all_np.extend(pred_np)
    # calculate off diagonal prediction
    if (config.n_class > 2):
        pred_offdia = pred.data.numpy() if not use_gpu else pred.data.cpu().numpy()
        for i in range(len(pred_offdia)):
            p = pred_offdia[i]
            t = target_np[i]
            if (np.abs(p - t) <= 1):
                pred_offdia[i] = target.data[i]
        pred_all_offdiagonal_np.extend(pred_offdia)

# Output confusion matrix
text_file = open(config.output_results, "w")
print("Confusion Matrix=\n", confusion_matrix)
text_file.write("Confusion Matrix=\n")
text_file.write(np.array2string(confusion_matrix.numpy()))

confusion_matrix_per = confusion_matrix / confusion_matrix.sum(1).view(config.n_class,1) #torch.FloatTensor(class_total)
print("\nConfusion Matrix Percentage=\n", confusion_matrix_per)
text_file.write("Confusion Matrix Percentage=\n")
text_file.write(np.array2string(confusion_matrix_per.numpy()))

"""print("")
print(confusion_matrix.diag()/confusion_matrix.sum(1))
print("")
TN= confusion_matrix[0, 0]
FP= confusion_matrix[0, 1]
FN= confusion_matrix[1, 0]
TP= confusion_matrix[1, 1]
print("TN=",TN,"FN=", FN," FP=", FP, "TP=", TP)"""


for i in range(config.n_class):
    if class_total[i] > 0:
        output_str = '\nTest Accuracy of %5s: %.2f%% (%2d/%2d)' % (
            config.classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i]))
    else:
        output_str = '\nTest Accuracy of %5s: N/A (no training examples)' % (config.classes[i])
    print(output_str)
    text_file.write(output_str)

output_str = '\nTest Accuracy (Overall): %.2f%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total))
print(output_str)
text_file.write(output_str)
print("")
print("Testing the Trained Network is Finished")

# AUC score
print("\nAUC:\n")
target_all_np = np.array(target_all_np).flatten()
pred_all_np = np.array(pred_all_np).flatten()
pred_all_offdiagonal_np = np.array(pred_all_offdiagonal_np).flatten()
if (config.n_class == 2):
    auc = roc_auc_score_multiclass(target_all_np, pred_all_np)
else:
    auc = roc_auc_score_multiclass(target_all_np, pred_all_np)
    auc_off = roc_auc_score_multiclass(target_all_np,pred_all_offdiagonal_np)
print(auc)
text_file.write('\nAUC:\n')
text_file.write(str(auc))

if (config.n_class > 2):
    print("\nAUC off_diagonal:\n")
    print(auc_off)
    text_file.write('\nAUC off_diagonal:\n')
    text_file.write(str(auc_off))

text_file.close()