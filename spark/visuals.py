import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
from pyspark.sql.functions import col, sum, when


from pyspark.sql import SparkSession

# ========== CONFIG ========== #
output_dir = "/opt/spark/output"
os.makedirs(output_dir, exist_ok=True)

# ========== Start Spark ========== #
spark = SparkSession.builder \
    .appName("VisualizationOnly") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ========== Load Training and Prediction Data ========== #
train_df = spark.read.csv("hdfs://namenode:8020/data/Training.csv", header=True, inferSchema=True)
predictions = spark.read.parquet("/opt/spark/output/predictions.parquet")

# ========== Confusion Matrix ========== #
y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# ========== Evaluation Metric Plot ========== #
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=["green", "blue", "orange", "red"])
plt.ylim(0, 1.1)
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")

for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(f"{output_dir}/evaluation_metrics.png")
plt.close()

# ========== Class Distribution ========== #
train_dist = train_df.groupBy("prognosis").count().toPandas()
plt.figure(figsize=(12, 6))
plt.bar(train_dist['prognosis'], train_dist['count'], color='skyblue')
plt.xticks(rotation=90)
plt.title("Class Distribution in Training Data")
plt.xlabel("Disease Prognosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png")
plt.close()

# ========== Done ========== #
print(f"Visuals saved in {output_dir}")
spark.stop()
