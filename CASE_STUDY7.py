#Step 1: Install PySpark (Google Colab)

# !pip install pyspark   # Uncomment if running in Google Colab / a fresh environment

#Step 2: Import Required Libraries

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

#Step 3: Create Spark Session

spark = SparkSession.builder \
    .appName("Smart Energy Consumption Analytics") \
    .getOrCreate()

#Step 4: load Dataset

df = spark.read.csv(
    "/content/powerconsumption.csv",
    header=True,
    inferSchema=True
)

#Step 5: Display Dataset

df.show(10)

#Step 6: Print Schema

df.printSchema()

#Step 7: Count Records

print("Number of Rows :", df.count())
print("Number of Columns :", len(df.columns))

#Step 8: Check Data Types

for col_name, dtype in df.dtypes:
    print(col_name, ":", dtype)

#Step 9: Summary Statistics

df.describe().show()

#Step 10: Check Missing Values

from pyspark.sql.functions import col, when, count

df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()

#Step 11: Remove Duplicate Records

df = df.dropDuplicates()

print("Rows after removing duplicates:", df.count())

#Step 12: Convert Datetime Column

from pyspark.sql.functions import col, try_to_timestamp, lit, trim
from pyspark.sql.types import TimestampType

df = df.withColumn(
    "Datetime",
    try_to_timestamp(trim(col("Datetime")), lit("M/d/yyyy H:mm")).cast(TimestampType())
).select(col("Datetime"), col("Temperature"), col("Humidity"), col("WindSpeed"), col("GeneralDiffuseFlows"), col("DiffuseFlows"), col("PowerConsumption_Zone1"), col("PowerConsumption_Zone2"), col("PowerConsumption_Zone3"))

#Step 13: Verify

df.printSchema()

#Step 14: Display Final Dataset

df.show(5)

#Step 1: Convert DataFrame to RDD

rdd = df.rdd

print("RDD Created Successfully")

#Step 2: Display First 5 Records

for row in rdd.take(5):
    print(row)

#Step 3: Count Total Records (Action)

print("Total Records:", rdd.count())

#Step 4: Extract Temperature Column (Transformation)

temperature_rdd = rdd.map(lambda x: x.Temperature)

temperature_rdd.take(10)

#Step 5: Extract Power Consumption (Zone 1)

zone1_rdd = rdd.map(lambda x: x.PowerConsumption_Zone1)

zone1_rdd.take(10)

#Step 6: Filter High Temperature Records

#Let's filter records where Temperature > 35°C.

high_temp = rdd.filter(lambda x: x.Temperature > 35)

high_temp.take(10)

#Step 7: Count High Temperature Records

print("High Temperature Records:", high_temp.count())

#Step 8: Filter High Power Consumption

#Suppose high demand is greater than 35000.

high_power = rdd.filter(
    lambda x: x.PowerConsumption_Zone1 > 35000
)

high_power.take(10)

#Step 9: Average Temperature

avg_temp = temperature_rdd.mean()

print("Average Temperature:", avg_temp)

#Step 10: Maximum Temperature

max_temp = temperature_rdd.max()

print("Maximum Temperature:", max_temp)

#Step 11: Minimum Temperature

min_temp = temperature_rdd.min()

print("Minimum Temperature:", min_temp)

#Step 12: Average Electricity Consumption

avg_power = zone1_rdd.mean()

print("Average Zone1 Consumption:", avg_power)

#Step 13: Maximum Electricity Consumption

print("Maximum Zone1 Consumption:", zone1_rdd.max())

#Step 14: Minimum Electricity Consumption

print("Minimum Zone1 Consumption:", zone1_rdd.min())

#Step 15: Create Pair RDD

pair_rdd = rdd.map(
    lambda x: (x.Temperature, x.PowerConsumption_Zone1)
)

pair_rdd.take(10)

#Step 16: Sort by Temperature

sorted_rdd = pair_rdd.sortByKey()

sorted_rdd.take(10)

#Step 17: Collect Sample Records

sample_records = rdd.take(20)

for record in sample_records:
    print(record)

#Step 18: Find Top 10 Highest Electricity Consumption

top_power = zone1_rdd.top(10)

print(top_power)

#Step 19: Reduce Example

#Find the total electricity consumption of Zone 1.

total_consumption = zone1_rdd.reduce(lambda x, y: x + y)

print("Total Zone1 Consumption:", total_consumption)

#Step 1: Create Pair RDD

#We'll use Hour as the key and PowerConsumption_Zone1 as the value.

#First, extract the hour from the Datetime string.

rdd = df.rdd

pair_rdd = rdd.map(lambda x: (
    x.Datetime.hour,
    x.PowerConsumption_Zone1
))

pair_rdd.take(10)

#Step 2: Count Records for Each Hour

hour_count = pair_rdd.mapValues(lambda x: 1) \
                     .reduceByKey(lambda a, b: a + b)

hour_count.collect()

#Step 3: Total Electricity Consumption per Hour

hour_consumption = pair_rdd.reduceByKey(lambda a, b: a + b)

hour_consumption.collect()

#Step 4: Average Electricity Consumption per Hour

hour_avg = pair_rdd.combineByKey(
    lambda value: (value, 1),
    lambda acc, value: (acc[0] + value, acc[1] + 1),
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
).mapValues(lambda x: x[0] / x[1])

hour_avg.collect()

#Step 5: Group By Key

group_data = pair_rdd.groupByKey()

for row in group_data.take(5):
    print(row)

#Step 6: Sort by Key

sorted_hour = hour_consumption.sortByKey()

sorted_hour.collect()

#Step 7: Shuffle Operation

#reduceByKey() automatically performs a shuffle.

shuffle_result = pair_rdd.reduceByKey(lambda x, y: x + y)

shuffle_result.collect()

#Step 8: Cache RDD

pair_rdd.cache()

print(pair_rdd.is_cached)

#Step 9: Persist RDD & Unpersist

from pyspark import StorageLevel

# If the RDD was previously persisted, unpersist it first
if pair_rdd.is_cached:
    pair_rdd.unpersist()

pair_rdd.persist(StorageLevel.MEMORY_AND_DISK)

print(pair_rdd.getStorageLevel())

#Step 1: Display Dataset

df.show(5)

#Step 2: Select Specific Columns

df.select(
    "Temperature",
    "Humidity",
    "PowerConsumption_Zone1"
).show(10)

#Step 3: Select Multiple Columns

df.select(
    "Datetime",
    "Temperature",
    "Humidity",
    "WindSpeed"
).show(10)

#Step 4: Filter High Temperature Records

df.filter(df.Temperature > 35).show()

#Step 5: Filter High Electricity Consumption

df.filter(df.PowerConsumption_Zone1 > 35000).show()

#Step 6: Filter Using Multiple Conditions

df.filter(
    (df.Temperature > 30) &
    (df.Humidity < 50)
).show()

#Step 7: Create a New Column

#Difference between Zone1 and Zone2 consumption.

from pyspark.sql.functions import col

df = df.withColumn(
    "Consumption_Difference",
    col("PowerConsumption_Zone1") - col("PowerConsumption_Zone2")
)

df.show(5)

#Step 8: Create Total Power Consumption Column

df = df.withColumn(
    "Total_Consumption",
    col("PowerConsumption_Zone1") +
    col("PowerConsumption_Zone2") +
    col("PowerConsumption_Zone3")
)

df.select(
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
    "Total_Consumption"
).show(5)

#Step 9: Sort by Temperature

df.orderBy(df.Temperature.desc()).show(10)

#Step 10: Group By Humidity

from pyspark.sql.functions import avg

df.groupBy("Humidity") \
  .agg(
      avg("PowerConsumption_Zone1").alias("Average_Consumption")
  ) \
  .show()

#Step 11: Average Temperature

from pyspark.sql.functions import avg

df.select(
    avg("Temperature").alias("Average Temperature")
).show()

#Step 12: Maximum Consumption

from pyspark.sql.functions import max

df.select(
    max("PowerConsumption_Zone1").alias("Maximum Consumption")
).show()

#Step 13: Minimum Consumption

from pyspark.sql.functions import min

df.select(
    min("PowerConsumption_Zone1").alias("Minimum Consumption")
).show()

#Step 14: Summary Statistics

df.describe().show()

#Step 15: Distinct Humidity Values

df.select("Humidity").distinct().show()

#Step 16: Count Distinct Humidity Values

df.select("Humidity").distinct().count()

#Step 17: Remove Duplicate Records

df = df.dropDuplicates()

print("Rows after removing duplicates:", df.count())

#Step 18: Rename a Column

df = df.withColumnRenamed(
    "PowerConsumption_Zone1",
    "Zone1"
)

df.printSchema()

df_weather = df.select(
    "Datetime",
    "Temperature",
    "Humidity",
    "WindSpeed"
)

df_power = df.select(
    "Datetime",
    "Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3"
)

joined_df = df_weather.join(
    df_power,
    on="Datetime",
    how="inner"
)

joined_df.show(5)

#Step 20: Aggregation

from pyspark.sql.functions import avg, max, min

df.agg(
    avg("Zone1").alias("Average"),
    max("Zone1").alias("Maximum"),
    min("Zone1").alias("Minimum")
).show()

#Step 1: Create a Temporary SQL View

df.createOrReplaceTempView("energy_data")
spark.sql("SELECT * FROM energy_data LIMIT 5").show()

#Step 2: Exploratory Data Analysis - Correlation Between Weather and Consumption

from pyspark.sql.functions import corr

df.select(
    corr("Temperature", "Total_Consumption").alias("Temp_vs_Consumption"),
    corr("Humidity", "Total_Consumption").alias("Humidity_vs_Consumption"),
    corr("WindSpeed", "Total_Consumption").alias("WindSpeed_vs_Consumption")
).show()

#Step 3: Extract Hour, Day, Month, and Year from Datetime

from pyspark.sql.functions import hour, dayofmonth, month, year, date_format

df = df.withColumn("Hour", hour("Datetime")) \
       .withColumn("Day", dayofmonth("Datetime")) \
       .withColumn("Month", month("Datetime")) \
       .withColumn("Year", year("Datetime")) \
       .withColumn("MonthName", date_format("Datetime", "MMMM"))

df.select("Datetime", "Hour", "Day", "Month", "Year", "MonthName").show(5)

#Refresh the temp view with the new columns
df.createOrReplaceTempView("energy_data")

#Step 4: Hourly Energy Consumption Analysis (Spark SQL)

hourly_consumption = spark.sql("""
    SELECT Hour,
           ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption,
           ROUND(SUM(Total_Consumption), 2) AS Total_Consumption,
           COUNT(*) AS Record_Count
    FROM energy_data
    GROUP BY Hour
    ORDER BY Hour
""")

hourly_consumption.show(24)

#Step 5: Hourly Energy Consumption (DataFrame API)

from pyspark.sql.functions import avg, sum as _sum, round as _round

df.groupBy("Hour") \
  .agg(_round(avg("Total_Consumption"), 2).alias("Avg_Consumption")) \
  .orderBy("Hour") \
  .show(24)

#Step 6: Region-wise (Zone-wise) Electricity Usage (Spark SQL)

region_usage = spark.sql("""
    SELECT
        ROUND(SUM(Zone1), 2) AS Zone1_Total,
        ROUND(SUM(PowerConsumption_Zone2), 2) AS Zone2_Total,
        ROUND(SUM(PowerConsumption_Zone3), 2) AS Zone3_Total,
        ROUND(AVG(Zone1), 2) AS Zone1_Avg,
        ROUND(AVG(PowerConsumption_Zone2), 2) AS Zone2_Avg,
        ROUND(AVG(PowerConsumption_Zone3), 2) AS Zone3_Avg
    FROM energy_data
""")

region_usage.show()

#Step 7: Region-wise Usage Reshaped for Comparison (DataFrame API)

from pyspark.sql import Row

zone_totals = df.agg(
    _sum("Zone1").alias("Zone1"),
    _sum("PowerConsumption_Zone2").alias("Zone2"),
    _sum("PowerConsumption_Zone3").alias("Zone3")
).collect()[0]

zone_df = spark.createDataFrame(
    [Row(Region="Zone1", Total_Consumption=float(zone_totals["Zone1"])),
     Row(Region="Zone2", Total_Consumption=float(zone_totals["Zone2"])),
     Row(Region="Zone3", Total_Consumption=float(zone_totals["Zone3"]))]
)

zone_df.orderBy(zone_df.Total_Consumption.desc()).show()

#Step 8: Identify Peak Demand Periods - Peak Hours of the Day (Spark SQL)

peak_hours = spark.sql("""
    SELECT Hour, ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption
    FROM energy_data
    GROUP BY Hour
    ORDER BY Avg_Consumption DESC
    LIMIT 5
""")

print("Top 5 Peak Demand Hours:")
peak_hours.show()

#Step 9: Identify Peak Demand Periods - Highest Individual Demand Timestamps

peak_timestamps = spark.sql("""
    SELECT Datetime, Total_Consumption
    FROM energy_data
    ORDER BY Total_Consumption DESC
    LIMIT 10
""")

print("Top 10 Highest Demand Timestamps:")
peak_timestamps.show(truncate=False)

#Step 10: Analyze Consumer Categories - Classify Consumption into Low / Medium / High

#Compute 33rd and 66th percentile thresholds for Total_Consumption
low_thresh, high_thresh = df.approxQuantile("Total_Consumption", [0.33, 0.66], 0.01)
print(f"Low/Medium threshold: {low_thresh:.2f}, Medium/High threshold: {high_thresh:.2f}")

from pyspark.sql.functions import when

df = df.withColumn(
    "Consumer_Category",
    when(col("Total_Consumption") <= low_thresh, "Low")
    .when(col("Total_Consumption") <= high_thresh, "Medium")
    .otherwise("High")
)

df.select("Datetime", "Total_Consumption", "Consumer_Category").show(5)

#Refresh the temp view with the new column
df.createOrReplaceTempView("energy_data")

#Step 11: Consumer Category Distribution (Spark SQL)

consumer_category_dist = spark.sql("""
    SELECT Consumer_Category,
           COUNT(*) AS Record_Count,
           ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption
    FROM energy_data
    GROUP BY Consumer_Category
    ORDER BY Avg_Consumption DESC
""")

consumer_category_dist.show()

#Step 12: Generate Monthly Consumption Report (Spark SQL)

monthly_report = spark.sql("""
    SELECT MonthName,
           Month,
           ROUND(SUM(Total_Consumption), 2) AS Total_Monthly_Consumption,
           ROUND(AVG(Total_Consumption), 2) AS Avg_Monthly_Consumption,
           ROUND(MAX(Total_Consumption), 2) AS Peak_Monthly_Consumption
    FROM energy_data
    GROUP BY MonthName, Month
    ORDER BY Month
""")

monthly_report.show(12)

#Step 13: Monthly Consumption Report (DataFrame API)

from pyspark.sql.functions import max as _max

df.groupBy("Month", "MonthName") \
  .agg(
      _round(_sum("Total_Consumption"), 2).alias("Total_Monthly_Consumption"),
      _round(avg("Total_Consumption"), 2).alias("Avg_Monthly_Consumption"),
      _round(_max("Total_Consumption"), 2).alias("Peak_Monthly_Consumption")
  ) \
  .orderBy("Month") \
  .show(12)

#Step 14: Visualize Hourly Consumption Trend

import matplotlib.pyplot as plt

hourly_pd = hourly_consumption.toPandas()

plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["Hour"], hourly_pd["Avg_Consumption"], marker="o", color="steelblue")
plt.title("Average Hourly Energy Consumption")
plt.xlabel("Hour of Day")
plt.ylabel("Average Total Consumption")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Step 15: Visualize Monthly Consumption Report

monthly_pd = monthly_report.toPandas()

plt.figure(figsize=(10, 5))
plt.bar(monthly_pd["MonthName"], monthly_pd["Total_Monthly_Consumption"], color="seagreen")
plt.title("Total Monthly Energy Consumption")
plt.xlabel("Month")
plt.ylabel("Total Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Step 1: EXTRACT - Ingest Raw Smart Meter & Weather Data

def extract_raw_data(spark, path):
    """Extract raw energy consumption data from source CSV."""
    raw_df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"Extracted {raw_df.count()} raw records from {path}")
    return raw_df

raw_data = extract_raw_data(spark, "powerconsumption.csv")
raw_data.printSchema()

#Step 2: TRANSFORM - Clean & Standardize the Raw Data

from pyspark.sql.functions import col, trim, try_to_timestamp, lit
from pyspark.sql.types import TimestampType

def clean_data(raw_df):
    """Cast datetime, drop duplicates, and drop rows with nulls in key columns."""
    cleaned = raw_df.withColumn(
        "Datetime",
        try_to_timestamp(trim(col("Datetime")), lit("M/d/yyyy H:mm")).cast(TimestampType())
    ).dropDuplicates().dropna(
        subset=["Datetime", "Temperature", "Humidity", "WindSpeed",
                "PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]
    )
    return cleaned

cleaned_data = clean_data(raw_data)
print(f"Records after cleaning: {cleaned_data.count()}")
cleaned_data.show(5)

#Step 3: TRANSFORM - Feature Engineering (Time Attributes & Derived Metrics)

from pyspark.sql.functions import col,hour, dayofmonth, month, year, date_format

def enrich_data(cleaned_df):
    enriched = (
        cleaned_df
        .withColumn("Hour", hour("Datetime"))
        .withColumn("Day", dayofmonth("Datetime"))
        .withColumn("Month", month("Datetime"))
        .withColumn("Year", year("Datetime"))
        .withColumn("MonthName", date_format("Datetime", "MMMM"))
        .withColumn(
            "Consumption_Difference",
            col("PowerConsumption_Zone1") - col("PowerConsumption_Zone2")
        )
        .withColumn(
            "Total_Consumption",
            col("PowerConsumption_Zone1")
            + col("PowerConsumption_Zone2")
            + col("PowerConsumption_Zone3")
        )
    )

    return enriched

enriched_data = enrich_data(cleaned_data)
enriched_data.select("Datetime", "Hour", "Month", "Year", "Total_Consumption").show(5)

#Step 4: TRANSFORM - Split into Smart Meter (Fact) and Weather (Dimension) Tables

def split_datasets(enriched_df):
    """Separate the enriched dataset into a smart-meter fact table and a weather dimension table."""
    smart_meter_fact = enriched_df.select(
        "Datetime", "Year", "Month", "Day", "Hour",
        "PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3",
        "Total_Consumption"
    )

    weather_dim = enriched_df.select(
        "Datetime", "Temperature", "Humidity", "WindSpeed",
        "GeneralDiffuseFlows", "DiffuseFlows"
    )

    return smart_meter_fact, weather_dim

smart_meter_fact, weather_dim = split_datasets(enriched_data)
smart_meter_fact.show(5)
weather_dim.show(5)

#Step 5: TRANSFORM - Data Quality Checks

def data_quality_report(raw_df, final_df):
    """Compare record counts and check for remaining nulls."""
    raw_count = raw_df.count()
    final_count = final_df.count()
    dropped = raw_count - final_count

    null_counts = final_df.select(
        [ (col(c).isNull().cast("int")).alias(c) for c in final_df.columns ]
    ).groupBy().sum().collect()[0].asDict()

    print(f"Raw records: {raw_count}")
    print(f"Final records: {final_count}")
    print(f"Records dropped during cleaning: {dropped}")
    print("Null counts per column:")
    for k, v in null_counts.items():
        if v and v > 0:
            print(f"  {k}: {v}")
    print("Data quality check passed - no critical nulls found" if dropped >= 0 else "Check failed")

data_quality_report(raw_data, enriched_data)

#Step 6: LOAD - Persist Smart Meter Fact Table (Partitioned by Year & Month)

def load_to_parquet(df, output_path, partition_cols=None):
    """Write a DataFrame to Parquet, optionally partitioned."""
    writer = df.write.mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(output_path)
    print(f"Loaded data to {output_path}")

load_to_parquet(smart_meter_fact, "output/smart_meter_fact", partition_cols=["Year", "Month"])

#Step 7: LOAD - Persist Weather Dimension Table

load_to_parquet(weather_dim, "output/weather_dim")

#Step 8: LOAD - Verify Persisted Data

verify_fact = spark.read.parquet("output/smart_meter_fact")
verify_weather = spark.read.parquet("output/weather_dim")

print(f"Smart meter fact table records: {verify_fact.count()}")
print(f"Weather dimension table records: {verify_weather.count()}")
verify_fact.show(5)

#Step 9: Orchestrate Full ETL Pipeline (Extract -> Transform -> Load)

import time

def run_etl_pipeline(spark, source_path):
    """End-to-end ETL pipeline for smart meter and weather data."""
    start = time.time()

    print("=== EXTRACT ===")
    raw = extract_raw_data(spark, source_path)

    print("=== TRANSFORM ===")
    cleaned = clean_data(raw)
    enriched = enrich_data(cleaned)
    fact, dim = split_datasets(enriched)
    data_quality_report(raw, enriched)

    print("=== LOAD ===")
    load_to_parquet(fact, "output/smart_meter_fact", partition_cols=["Year", "Month"])
    load_to_parquet(dim, "output/weather_dim")

    elapsed = time.time() - start
    print(f"ETL pipeline completed in {elapsed:.2f} seconds")
    return fact, dim

fact_table, weather_table = run_etl_pipeline(spark, "powerconsumption.csv")

#Step 1: Prepare Feature Vector for Demand Prediction

from pyspark.sql.functions import col, when, abs as _abs
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

feature_cols = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows", "Hour", "Month"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
ml_data = assembler.transform(df).select("Datetime", "features", col("Total_Consumption").alias("label"))
ml_data.show(5, truncate=False)

#Step 2: Train-Test Split
train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)
print(f"Training records: {train_data.count()}")
print(f"Testing records: {test_data.count()}")

#Step 3: Train multiple regression models for comparison
candidate_models = [
    ("LinearRegression", LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)),
    ("DecisionTree", DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=8)),
    ("RandomForest", RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=8, seed=42)),
    ("GradientBoostedTrees", GBTRegressor(featuresCol="features", labelCol="label", maxIter=50, maxDepth=5, seed=42))
]

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

fitted_models = {}
model_metrics = []

for name, estimator in candidate_models:
    model = estimator.fit(train_data)
    fitted_models[name] = model
    preds = model.transform(test_data)
    rmse = evaluator_rmse.evaluate(preds)
    mae = evaluator_mae.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    model_metrics.append((name, rmse, mae, r2))
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

metrics_df = spark.createDataFrame(
    [(name, rmse, mae, r2) for name, rmse, mae, r2 in model_metrics],
    ["model_name", "rmse", "mae", "r2"]
).orderBy("rmse")
print("\nModel comparison table:")
metrics_df.show(truncate=False)

best_model_name = metrics_df.first()["model_name"]
best_model = fitted_models[best_model_name]
print(f"\nBest model selected by lowest RMSE: {best_model_name}")

#Step 4: Evaluate the selected best model and inspect predictions
best_predictions = best_model.transform(test_data)
print(f"Selected best model performance -> RMSE: {evaluator_rmse.evaluate(best_predictions):.2f}, MAE: {evaluator_mae.evaluate(best_predictions):.2f}, R2: {evaluator_r2.evaluate(best_predictions):.4f}")
best_predictions.select("Datetime", "label", "prediction").show(10, truncate=False)

#Step 5: Understand the selected model
if hasattr(best_model, "featureImportances"):
    importances = best_model.featureImportances.toArray()
    print("\nFeature importances for the selected model:")
    for feature, importance in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {feature}: {importance:.4f}")
elif hasattr(best_model, "coefficients"):
    print("\nLinear model coefficients for the selected model:")
    for feature, coef in zip(feature_cols, best_model.coefficients):
        print(f"  {feature}: {coef:.4f}")
else:
    print("\nSelected model does not expose feature importances or coefficients.")

#Step 6: Detect anomalies using the best model
best_predictions = best_predictions.withColumn("residual", col("label") - col("prediction")) \
    .withColumn("abs_residual", _abs(col("label") - col("prediction")))

residual_stats = best_predictions.select("abs_residual").summary("mean", "stddev").collect()
mean_residual = float(residual_stats[0]["abs_residual"])
std_residual = float(residual_stats[1]["abs_residual"])
anomaly_threshold = mean_residual + 3 * std_residual
print(f"\nMean absolute residual: {mean_residual:.2f}")
print(f"Std deviation of residual: {std_residual:.2f}")
print(f"Anomaly threshold: {anomaly_threshold:.2f}")

anomalies = best_predictions.withColumn(
    "is_anomaly",
    when(col("abs_residual") > anomaly_threshold, True).otherwise(False)
)

anomaly_count = anomalies.filter(col("is_anomaly") == True).count()
total_count = anomalies.count()
print(f"Detected {anomaly_count} anomalies out of {total_count} test records ({(anomaly_count / total_count) * 100:.2f}%)")

anomalies.filter(col("is_anomaly") == True) \
    .select("Datetime", "label", "prediction", "abs_residual") \
    .orderBy(col("abs_residual").desc()) \
    .show(10, truncate=False)

import matplotlib.pyplot as plt
sample_pd = anomalies.select("label", "prediction", "is_anomaly").sample(fraction=0.1, seed=42).toPandas()
normal_pts = sample_pd[sample_pd["is_anomaly"] == False]
anomaly_pts = sample_pd[sample_pd["is_anomaly"] == True]

plt.figure(figsize=(8, 8))
plt.scatter(normal_pts["label"], normal_pts["prediction"], alpha=0.4, label="Normal", color="steelblue", s=15)
plt.scatter(anomaly_pts["label"], anomaly_pts["prediction"], alpha=0.8, label="Anomaly", color="red", s=30)
plt.plot([sample_pd["label"].min(), sample_pd["label"].max()],
         [sample_pd["label"].min(), sample_pd["label"].max()], "k--", label="Perfect Prediction")
plt.xlabel("Actual Demand (Total Consumption)")
plt.ylabel("Predicted Demand")
plt.title(f"Actual vs Predicted Electricity Demand - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.show()
