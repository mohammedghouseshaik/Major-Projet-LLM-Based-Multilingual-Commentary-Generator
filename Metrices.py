import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Load Ground Truth and Generated Commentary
with open("reference_commentary.json", "r", encoding="utf-8") as f:
    reference_commentary = json.load(f)

with open("generated_commentary.json", "r", encoding="utf-8") as f:
    generated_commentary = json.load(f)

# Flatten generated_commentary if it contains nested lists
if isinstance(generated_commentary[0], list):
    generated_commentary = [item for sublist in generated_commentary for item in sublist]

# Ensure equal number of samples
min_length = min(len(generated_commentary), len(reference_commentary))

# Trim both lists to match lengths
generated_commentary = generated_commentary[:min_length]
reference_commentary = reference_commentary[:min_length]

# Extract commentary for evaluation
y_pred = [entry["commentary"]["Traditional Ball-by-Ball"] for entry in generated_commentary]
y_true = [entry["commentary"]["Traditional Ball-by-Ball"] for entry in reference_commentary]

# Compute BERTScore
P, R, F1 = bert_score(y_pred, y_true, lang="en", rescale_with_baseline=True)

# Print BERTScore Metrics
print("BERT Precision: ", np.mean(P.numpy()))
print("BERT Recall: ", np.mean(R.numpy()))
print("BERT F1-Score: ", np.mean(F1.numpy()))

# Compute ROUGE Scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(pred, true) for pred, true in zip(y_pred, y_true)]

rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")

# Simulate Binary Classification Labels (1 for Good Commentary, 0 for Bad Commentary)
labels = ["Good Commentary", "Bad Commentary"]
y_true_labels = [1 if "good" in ref.lower() else 0 for ref in y_true]  # Simple rule-based labeling
y_pred_labels = [1 if "good" in pred.lower() else 0 for pred in y_pred]

# Generate Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Normalized Confusion Matrix
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true_labels, y_pred_labels)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# F1 Score Curve
plt.figure(figsize=(6, 5))
plt.plot(F1.numpy(), label="F1 Score Curve")
plt.xlabel("Samples")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 Score Curve")
plt.show()
