from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, sum, when
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("DiseasePredictionModel") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df_train = spark.read.csv("hdfs://namenode:8020/data/Training.csv", header=True, inferSchema=True)
df_test = spark.read.csv("hdfs://namenode:8020/data/Testing.csv", header=True, inferSchema=True)

# Combine them
df = df_train.unionByName(df_test)

df.printSchema()
df.select("itching","skin_rash","nodal_skin_eruptions","chills","prognosis").show()

print("\nColumns: ",df.columns)

print("\nSize of each Classes: ")
df.groupby("prognosis").count().show(1000, truncate=False)

print("\nChecking null values: ")
df.select([sum(when((col(c) != 0) | (col(c) != 1), 0).otherwise(1)).alias(c) for c in df.columns]).show()

print(f"\nRows: {df.count()}, Columns: {len(df.columns)}")

# Get all symptom columns (excluding 'prognosis')
symptom_cols = [c for c in df.columns if c != 'prognosis']

# Sum each symptom column
symptom_counts = df.select([
    sum(col(c)).alias(c + "_count") for c in symptom_cols
])
print("\nSymptom Frequency")
# Convert to Pandas
symptom_counts_pd = symptom_counts.toPandas()

# Transpose and sort
symptom_freq_sorted = symptom_counts_pd.T
symptom_freq_sorted.columns = ['count']
symptom_freq_sorted = symptom_freq_sorted.sort_values(by='count', ascending=False)

print(symptom_freq_sorted)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

label_indexer = StringIndexer(inputCol="prognosis", outputCol="label")
label_indexer_model = label_indexer.fit(df) 
cdf = label_indexer_model.transform(df)
print("\nPrognosis and their corresponding label: ")
cdf.select("prognosis", "label").distinct().orderBy("label").show(1000, truncate=False)

symptom_cols = df.columns[:-2] 
assembler = VectorAssembler(inputCols=symptom_cols, outputCol="features")
# df = assembler.transform(df).select("features", "label")

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
pipeline = Pipeline(stages=[label_indexer, assembler, rf])
model = pipeline.fit(train_df)

                                        ########### #######  ########  ##########
                                            ##      ##       ##            ##
                                            ##      #######  ########      ##
                                            ##      ##             ##      ##
                                            ##      #######  ########      ##

predictions = model.transform(test_df)
print("\nPredictions: ")
predictions.select("prognosis", "label", "prediction").show()

# Define evaluator for accuracy, precision, recall, and F1-score
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Accuracy
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print(f"\nAccuracy: {accuracy:.2f}")

# Precision
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print(f"\nPrecision: {precision:.2f}")

# Recall
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print(f"\nRecall: {recall:.2f}")

# F1-Score
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f"\nF1 Score: {f1:.2f}")

model.save("rf_pipeline_model")
predictions.select("label", "prediction").write.mode("overwrite").parquet("/opt/spark/output/predictions.parquet")


spark.stop()
