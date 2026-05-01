#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# import kagglehub
# import os


# In[3]:


# path = kagglehub.dataset_download("mckenziemakwela/synthetic-global-bank-transactions-dataset")
# print("Path to dataset files:", path)


# In[4]:


# files = os.listdir(path)
# print(files)


# # IMPORT LIBRARIES

# In[5]:


import findspark
findspark.init()


# In[6]:


import os
from functools import reduce
from pyspark.sql import SparkSession, Window, Row # type: ignore
from pyspark.ml.linalg import Vectors # type: ignore
from pyspark.sql.functions import * # type: ignore
from pyspark.ml.feature import OneHotEncoder, StandardScalerModel, StandardScaler, VectorAssembler, StringIndexer # type: ignore
from pyspark.ml.classification import LogisticRegressionModel # type: ignore
from pyspark.ml import Pipeline, PipelineModel # type: ignore
from pyspark.sql.types import DoubleType # type: ignore
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier # type: ignore
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator # type: ignore
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder # type: ignore
import shutil
import json
import builtins


# # INITIALIZE SPARK SESSION

# In[7]:


spark = SparkSession.builder \
    .appName("Bank Dataset") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.log.level", "ERROR") \
    .config("log4j.logLevel", "ERROR") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# # LOAD DATA

# In[8]:


# customers_df = spark.read.csv(
#     f"{path}/customers.csv",
#     header=True,
#     inferSchema=True
# )

# transactions_df = spark.read.csv(
#     f"{path}/transactions.csv",
#     header=True,
#     inferSchema=True
# )


# In[9]:


customers_df = spark.read.csv(
    r"DataSet\customers.csv",
    header=True,
    inferSchema=True
)

transactions_df = spark.read.csv(
    r"DataSet\transactions.csv",
    header=True,
    inferSchema=True
)


# # JOIN DATASETS

# In[10]:


print("\nJoining datasets...")
df = transactions_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)
df.show(50)

original_count = df.count()
print(f"Original data size before any processing: {original_count:,}")


# # SHOW THE SCHEMA

# In[11]:


[schema.name for schema in df.schema]


# # FEATURE ENGINEERING

# In[12]:


df = df.withColumn(
    "transaction_datetime", 
    to_timestamp(col("transaction_timestamp")) # type: ignore
)

df = df.withColumn("hour", hour(col("transaction_datetime"))) # type: ignore
df = df.withColumn("day_of_week", dayofweek(col("transaction_datetime"))) # type: ignore

percentiles = df.approxQuantile("amount", [0.25, 0.5, 0.75], 0.05)
df = df.withColumn(
    "amount_category",
    when(col("amount") <= percentiles[0], "small") # type: ignore
    .when(col("amount") <= percentiles[1], "medium") # type: ignore
    .when(col("amount") <= percentiles[2], "large") # type: ignore
    .otherwise("very_large")
)


# # CALCULATE HISTORICAL FRAUD RATES

# In[13]:


window_spec = Window.partitionBy("customer_id").orderBy("transaction_datetime")

df = df.withColumn("customer_fraud_rate",
    when(count("is_fraud").over(window_spec.rowsBetween(-100, -1)) > 0, # type: ignore
         (sum("is_fraud").over(window_spec.rowsBetween(-100, -1)) /  
          count("is_fraud").over(window_spec.rowsBetween(-100, -1))) # type: ignore
    ).otherwise(0.0)
)

df = df.withColumn("customer_transaction_count",
    count("*").over(window_spec.rowsBetween(Window.unboundedPreceding, 0)) # type: ignore
)

df = df.withColumn("customer_fraud_count",
    sum("is_fraud").over(window_spec.rowsBetween(Window.unboundedPreceding, 0))
)

merchant_fraud_stats = df.groupBy("merchant_category").agg(
    avg(col("is_fraud")).alias("merchant_fraud_rate"), # type: ignore
    stddev(col("is_fraud")).alias("merchant_fraud_volatility") # type: ignore
)

df = df.join(merchant_fraud_stats, on="merchant_category", how="left")
df = df.fillna(0, subset=["merchant_fraud_rate", "merchant_fraud_volatility"])


# # OUTLIER DETECTION AND REMOVAL

# In[14]:


lower_bound = df.approxQuantile("amount", [0.005], 0.01)[0]  # 0.5th percentile
upper_bound = df.approxQuantile("amount", [0.995], 0.01)[0]  # 99.5th percentile

print(f"Capping bounds: ${lower_bound:.2f} - ${upper_bound:.2f}")

outlier_count = df.filter((col("amount") < lower_bound) | (col("amount") > upper_bound)).count() # type: ignore

df = df.withColumn("amount_original", col("amount")) # type: ignore
df = df.withColumn("amount",
    when(col("amount") < lower_bound, lower_bound) # type: ignore
    .when(col("amount") > upper_bound, upper_bound) # type: ignore
    .otherwise(col("amount")) # type: ignore
)

df = df.withColumn("was_outlier",
    when((col("amount_original") < lower_bound) | (col("amount_original") > upper_bound), 1) # type: ignore
    .otherwise(0))

print(f"Capped {outlier_count} outlier values")


# # ENCODING CATEGORICAL DATA

# In[15]:


indexer = StringIndexer(
    inputCol="merchant_category",
    outputCol="merchant_category_index"
)

encoder = OneHotEncoder(
    inputCol="merchant_category_index",
    outputCol="merchant_category_encoded"
)

pipeline_encoding = Pipeline(stages=[indexer, encoder])
df = pipeline_encoding.fit(df).transform(df)

df = df.withColumn("is_fraud_int", col("is_fraud").cast(DoubleType())) # type: ignore


# # FEATURE SELECTION

# In[16]:


feature_columns = [
    "amount",
    "hour",
    "day_of_week",
    "customer_fraud_rate",
    "merchant_fraud_rate",
    "customer_transaction_count",
    "customer_fraud_count",
    "merchant_fraud_volatility"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features_raw"
)
df = assembler.transform(df)


# # FEATURE SCALING

# In[17]:


scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df)
df = scaler_model.transform(df)


# # CLASS IMBALANCE TREATMENT

# In[18]:


print("\n" + "="*60)
print("CLASS IMBALANCE TREATMENT")
print("="*60)

initial_counts = df.groupBy("is_fraud_int").count().collect()
fraud_count = next((row["count"] for row in initial_counts if row["is_fraud_int"] == 1.0), 0)
non_fraud_count = next((row["count"] for row in initial_counts if row["is_fraud_int"] == 0.0), 0)

print(f"Initial: Non-Fraud={non_fraud_count:,}, Fraud={fraud_count:,}")

TARGET_SIZE = 250000

if non_fraud_count > TARGET_SIZE:
    print(f"Sampling non-fraud down to {TARGET_SIZE:,}")
    non_fraud_df = df.filter(col("is_fraud_int") == 0.0).sample(False, TARGET_SIZE / non_fraud_count, seed=42) # type: ignore
else:
    non_fraud_df = df.filter(col("is_fraud_int") == 0.0) # type: ignore

if fraud_count < TARGET_SIZE:
    print(f"Sampling fraud UP to {TARGET_SIZE:,} (with replacement)")
    sample_fraction = TARGET_SIZE / fraud_count
    fraud_df = df.filter(col("is_fraud_int") == 1.0).sample(True, sample_fraction, seed=42) # type: ignore

    current_fraud = fraud_df.count()
    if current_fraud > TARGET_SIZE:
        fraud_df = fraud_df.limit(TARGET_SIZE)
else:
    fraud_df = df.filter(col("is_fraud_int") == 1.0) # type: ignore

df_balanced = non_fraud_df.union(fraud_df)

df_balanced = df_balanced.withColumn("random", rand(seed=42)) # type: ignore
df_balanced = df_balanced.orderBy("random").drop("random")
df_balanced = df_balanced.cache()

final_counts = df_balanced.groupBy("is_fraud_int").count().collect()
final_fraud = next((row["count"] for row in final_counts if row["is_fraud_int"] == 1.0), 0)
final_non_fraud = next((row["count"] for row in final_counts if row["is_fraud_int"] == 0.0), 0)

print(f"\nAfter balancing:")
print(f"  Non-Fraud: {final_non_fraud:,}")
print(f"  Fraud: {final_fraud:,}")
print(f"  Total: {final_non_fraud + final_fraud:,}")

class_weights = {
    0.0: 1.0,
    1.0: non_fraud_count / fraud_count
}

df_balanced = df_balanced.withColumn(
    "class_weight",
    when(col("is_fraud_int") == 0.0, class_weights[0.0]) # type: ignore
    .otherwise(class_weights[1.0])
)


# # CREATE FINAL DATAFRAME

# In[19]:


final_df = df_balanced.select(
    col("customer_id"), # type: ignore
    col("transaction_id"), # type: ignore
    col("amount"), # type: ignore
    col("merchant_category"), # type: ignore
    col("hour"), # type: ignore
    col("day_of_week"), # type: ignore
    col("amount_category"), # type: ignore
    col("customer_fraud_rate"), # type: ignore
    col("merchant_fraud_rate"), # type: ignore
    col("customer_transaction_count"), # type: ignore
    col("customer_fraud_count"), # type: ignore
    col("merchant_fraud_volatility"), # type: ignore
    col("features_scaled"), # type: ignore
    col("is_fraud_int").alias("target"), # type: ignore
    col("class_weight") # type: ignore
)


# # FINAL STATISTICS

# In[20]:


print(f"Total rows: {final_df.count()}")
print(f"Number of columns: {len(final_df.columns)}")

fraud_stats = final_df.groupBy("target").count()
fraud_stats.show()

print("\nAmount statistics:")
final_df.select("amount").describe().show()

print("\nFraud rate statistics:")
final_df.select("customer_fraud_rate", "merchant_fraud_rate").describe().show()


# # SAVE PROCESSED DATA

# In[21]:


[schema.name for schema in final_df.schema]


# In[22]:


final_df.write.mode("overwrite").parquet("processed_fraud_data_balanced.parquet")
print("Data saved to: processed_fraud_data_balanced.parquet")


# # PREPARE DATA FOR MODEL TRAINING

# In[23]:


df_ready = final_df.select(
    col("features_scaled").alias("features"), # type: ignore
    col("target").alias("label"), # type: ignore
    col("class_weight") # type: ignore
)

print("\n=== PREPARED DATA ===")
df_ready.printSchema()
df_ready.show(5, truncate=False)

print("\n=== CLASS DISTRIBUTION AFTER BALANCING ===")
df_ready.groupBy("label").count().show()


# # TRAIN/TEST SPLIT

# In[24]:


df_ready_shuffled = df_ready.orderBy(rand(seed=42)) # type: ignore

train_df, test_df = df_ready_shuffled.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {train_df.count():,}")
print(f"Test set size: {test_df.count():,}")

print("\nTraining set class distribution:")
train_df.groupBy("label").count().show()

print("Test set class distribution:")
test_df.groupBy("label").count().show()


# # MODEL TRAINING

# In[25]:


models = {
    "Random Forest": RandomForestClassifier(
        featuresCol="features", 
        labelCol="label", 
        weightCol="class_weight",
        numTrees=150,
        maxDepth=12, 
        minInstancesPerNode=10,
        featureSubsetStrategy="auto",
        seed=42
    ),

    "Gradient Boosted Trees": GBTClassifier(
        featuresCol="features", 
        labelCol="label", 
        weightCol="class_weight",
        maxIter=100,
        maxDepth=6,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42
    ),

    "Logistic Regression": LogisticRegression(
        featuresCol="features", 
        labelCol="label", 
        weightCol="class_weight",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.5,
        standardization=True
    )
}

trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    print("-" * 40)

    try:
        trained_models[name] = model.fit(train_df)
        print(f"✓ {name} trained successfully!")
    except Exception as e:
        print(f"✗ Error training {name}: {str(e)}")


# # MODEL EVALUATION

# In[26]:


evaluators = {
    "F1-Score": MulticlassClassificationEvaluator(labelCol="label", metricName="f1", metricLabel=1.0),
    "Precision": MulticlassClassificationEvaluator(labelCol="label", metricName="precisionByLabel", metricLabel=1.0),
    "Recall": MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel", metricLabel=1.0),
    "Accuracy": MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
}

auc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

results = {}

for name, model in trained_models.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print('='*50)

    predictions = model.transform(test_df)

    results[name] = {}
    for metric_name, evaluator in evaluators.items():
        results[name][metric_name] = evaluator.evaluate(predictions)

    results[name]["AUC-ROC"] = auc_evaluator.evaluate(predictions)

    print(f"F1-Score:  {results[name]['F1-Score']:.4f}  ← Primary metric")
    print(f"Precision: {results[name]['Precision']:.4f}")
    print(f"Recall:    {results[name]['Recall']:.4f}")
    print(f"Accuracy:  {results[name]['Accuracy']:.4f}")
    print(f"AUC-ROC:   {results[name]['AUC-ROC']:.4f}")

    print("\nConfusion Matrix:")
    cm = predictions.groupBy("label", "prediction").count().collect()
    cm_dict = {(row["label"], row["prediction"]): row["count"] for row in cm}

    tn = cm_dict.get((0.0, 0.0), 0)
    fp = cm_dict.get((0.0, 1.0), 0)
    fn = cm_dict.get((1.0, 0.0), 0)
    tp = cm_dict.get((1.0, 1.0), 0)

    print(f"                     Predicted")
    print(f"                  Fraud      Legit")
    print(f"Actual Fraud     {tp:6d}    {fn:6d}")
    print(f"       Legit     {fp:6d}    {tn:6d}")

    if results[name]['F1-Score'] > 0.7:
        print(f"Good performance for fraud detection")
    elif results[name]['F1-Score'] > 0.5:
        print(f"Acceptable performance, consider tuning")
    else:
        print(f"Needs improvement, try different features")


# # FIND BEST MODEL
# 

# In[27]:


if results:
    best_model_name = builtins.max(results, key=lambda x: results[x]["F1-Score"])
    print(f"\n{'='*60}")
    print(f"BEST MODEL (by F1-Score): {best_model_name}")
    print(f"  F1-Score: {results[best_model_name]['F1-Score']:.4f}")
    print(f"  Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    print(f"  Precision: {results[best_model_name]['Precision']:.4f}")
    print(f"  Recall: {results[best_model_name]['Recall']:.4f}")
    print(f"  AUC-ROC: {results[best_model_name]['AUC-ROC']:.4f}")
    print('='*60)

    best_auc_name = builtins.max(results, key=lambda x: results[x]["AUC-ROC"])
    print(f"\nBEST MODEL (by AUC-ROC): {best_auc_name}")
    print(f"  AUC-ROC: {results[best_auc_name]['AUC-ROC']:.4f}")


# # SAVE ALL MODELS AND PREPROCESSORS FOR STREAMLIT

# In[28]:


os.makedirs("models", exist_ok=True)

for name, model in trained_models.items():
    model_path = f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    model.save(model_path)
    print(f"Saved {name} to {model_path}")

if results:
    best_model = trained_models[best_model_name]
    if os.path.exists("best_fraud_model"):
        shutil.rmtree("best_fraud_model")
    best_model.save("best_fraud_model")
    print(f"Saved best model to: best_fraud_model")

print("\nSaving preprocessing components...")

if os.path.exists("models/assembler"):
    shutil.rmtree("models/assembler")
assembler.save("models/assembler")
print("Assembler saved")

if os.path.exists("models/scaler"):
    shutil.rmtree("models/scaler")
scaler_model.save("models/scaler")
print("Scaler saved")

preprocessing_config = {
    "percentiles": percentiles,
    "lower_bound": float(lower_bound),
    "upper_bound": float(upper_bound),
    "feature_columns": feature_columns,
    "class_weights": class_weights,
    "target_ratio": 0.5,
    "was_outlier_feature": True,
    "capping_percentiles": [0.005, 0.995]
}

with open("models/preprocessing_config.json", "w") as f:
    json.dump(preprocessing_config, f, indent=2)
print("Preprocessing config saved")

model_info = {
    "best_model_name": best_model_name if results else "Logistic Regression (Weighted)",
    "best_by_auc": best_auc_name if results else "Random Forest (Weighted)",
    "all_models": list(trained_models.keys()),
    "metrics": results,
    "feature_columns": feature_columns,
    "class_weights": class_weights,
    "training_date": str(spark.sql("SELECT current_timestamp()").collect()[0][0])
}

with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model info saved")

merchant_categories = df.select("merchant_category").distinct().collect()
merchant_list = [row.merchant_category for row in merchant_categories]
with open("models/merchant_categories.json", "w") as f:
    json.dump(merchant_list, f, indent=2)
print("Merchant categories saved")


# # FEATURE IMPORTANCE (for tree-based models)

# In[29]:


print("\nFeature Importance Analysis:")
for name, model in trained_models.items():
    if hasattr(model, "featureImportances"):
        importances = model.featureImportances.toArray().tolist()
        feature_importance = list(zip(feature_columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 features for {name}:")
        for feat, imp in feature_importance[:5]:
            print(f"  {feat}: {imp:.4f}")


# # TEST PREDICTION FUNCTION

# In[30]:


# def test_prediction():
#     """Test function to verify saved components work"""

#     test_assembler = VectorAssembler.load("models/assembler")
#     test_scaler = StandardScalerModel.load("models/scaler")

#     from pyspark.ml.classification import GBTClassificationModel
#     test_model = GBTClassificationModel.load("best_fraud_model")

#     test_features = Vectors.dense([1500.0, 14, 4, 0.0, 0.03, 1, 0, 0.01])
#     test_df = spark.createDataFrame([Row(features=test_features)])

#     result = test_model.transform(test_df)
#     prediction = result.select("prediction").collect()[0][0]

#     print(f"\nTest prediction result: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
#     return prediction

# test_prediction()


# # SUMMARY REPORT

# In[31]:


print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Original data size: {original_count:,}")
print(f"After outlier capping: {df.count():,} (0 rows removed)")
print(f"After imbalance treatment: {final_df.count():,}")
print(f"Features engineered: {len(feature_columns)}")
print(f"Models trained: {len(trained_models)}")
if results:
    print(f"Best model (F1): {best_model_name}")
    print(f"Best F1-Score: {results[best_model_name]['F1-Score']:.4f}")
    print(f"Best AUC-ROC: {results[best_auc_name]['AUC-ROC']:.4f}" if 'best_auc_name' in locals() else "N/A")
print("="*60)


# # CLEANUP

# In[32]:


spark.stop()
print("\nTraining complete! All artifacts saved to 'models/' directory.")


# In[33]:


# Create a zip archive of your working directory
# !zip -r /kaggle/working/my_project_outputs.zip /kaggle/working/

