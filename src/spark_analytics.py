import os
import sys
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, max as _max, min as _min, sum as _sum,
    round as _round, when, try_to_timestamp, lit, trim,
    hour, dayofmonth, month, year, date_format, corr, abs as _abs
)
from pyspark.sql.types import TimestampType, Row
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def init_spark():
    """Q1: Initialize Spark Session."""
    print("--- Q1: Spark Session Initialization ---")
    spark = SparkSession.builder \
        .appName("Smart Energy Consumption Analytics") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session initialized successfully.")
    return spark

def load_dataset(spark, file_path):
    """Q1: Load energy consumption dataset."""
    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        # Fallback to parent directory if not found in local path
        parent_path = os.path.join("..", file_path)
        if os.path.exists(parent_path):
            file_path = parent_path
            print(f"File not found locally. Fallback to parent path: {file_path}")
        else:
            raise FileNotFoundError(f"Could not find dataset {file_path} in current or parent directory.")
            
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Initial data cleaning (Datetime parsing and drop duplicate records)
    df = df.dropDuplicates()
    df = df.withColumn(
        "Datetime",
        try_to_timestamp(trim(col("Datetime")), lit("M/d/yyyy H:mm")).cast(TimestampType())
    ).dropna(subset=["Datetime"])
    
    # Rename PowerConsumption_Zone1 to Zone1 to align with Q4/Q5 naming requirements
    df = df.withColumnRenamed("PowerConsumption_Zone1", "Zone1")
    
    print(f"Loaded {df.count()} rows and {len(df.columns)} columns.")
    return df

def run_rdd_operations(spark, df):
    """Q2: RDD Implementation."""
    print("\n--- Q2: RDD Implementation ---")
    rdd = df.rdd
    print("RDD Created successfully.")
    
    # Step 2: Take 5 records
    print("First 5 records in RDD:")
    for row in rdd.take(5):
        print(row)
        
    # Step 3: Count total records
    print(f"Total Records in RDD: {rdd.count()}")
    
    # Step 4: Extract Temperature Column
    temperature_rdd = rdd.map(lambda x: x.Temperature)
    print("Sample temperatures:", temperature_rdd.take(10))
    
    # Step 5: Extract Zone 1 power consumption
    zone1_rdd = rdd.map(lambda x: x.Zone1)
    print("Sample Zone1 consumption:", zone1_rdd.take(10))
    
    # Step 6 & 7: Filter High Temperature (>35°C) and count
    high_temp = rdd.filter(lambda x: x.Temperature > 35)
    print("Sample High Temp (>35°C) records:")
    for row in high_temp.take(3):
        print(row)
    print("High Temperature Records Count:", high_temp.count())
    
    # Step 8: Filter High Power Consumption (>35000)
    high_power = rdd.filter(lambda x: x.Zone1 > 35000)
    print("High Power Consumption Records Count (>35000):", high_power.count())
    
    # Step 9-11: RDD Temperature Stats
    print("Average Temperature (RDD):", temperature_rdd.mean())
    print("Maximum Temperature (RDD):", temperature_rdd.max())
    print("Minimum Temperature (RDD):", temperature_rdd.min())
    
    # Step 12-14: RDD Zone1 stats
    print("Average Zone1 Consumption (RDD):", zone1_rdd.mean())
    print("Maximum Zone1 Consumption (RDD):", zone1_rdd.max())
    print("Minimum Zone1 Consumption (RDD):", zone1_rdd.min())
    
    # Step 15 & 16: Create Pair RDD and Sort by Temperature
    pair_rdd = rdd.map(lambda x: (x.Temperature, x.Zone1))
    sorted_rdd = pair_rdd.sortByKey()
    print("Sample sorted by Temperature (Key, Value):", sorted_rdd.take(5))
    
    # Step 18: Find Top 10 highest consumption
    top_power = zone1_rdd.top(10)
    print("Top 10 highest power consumptions:", top_power)
    
    # Step 19: Reduce example (total consumption of Zone1)
    total_consumption = zone1_rdd.reduce(lambda x, y: x + y)
    print("Total Zone1 Consumption (RDD reduce):", total_consumption)
    
    return rdd

def run_key_value_operations(rdd):
    """Q3: Key-Value Operations and Persistence."""
    print("\n--- Q3: Key-Value Operations & Persistence ---")
    
    # Step 1: Create Pair RDD (Hour, Zone1)
    pair_rdd = rdd.map(lambda x: (x.Datetime.hour, x.Zone1))
    print("Sample Hour-Zone1 Pair RDD:", pair_rdd.take(5))
    
    # Step 2: Count records per Hour
    hour_count = pair_rdd.mapValues(lambda x: 1).reduceByKey(lambda a, b: a + b)
    print("Record Count per Hour (sorted by hour):", hour_count.sortByKey().collect())
    
    # Step 3: Total Electricity Consumption per Hour
    hour_consumption = pair_rdd.reduceByKey(lambda a, b: a + b)
    print("Total Consumption per Hour (sorted):", hour_consumption.sortByKey().collect())
    
    # Step 4: Average Electricity Consumption per Hour (combineByKey)
    # createCombiner, mergeValue, mergeCombiners
    hour_avg = pair_rdd.combineByKey(
        lambda value: (value, 1),
        lambda acc, value: (acc[0] + value, acc[1] + 1),
        lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
    ).mapValues(lambda x: x[0] / x[1])
    print("Average Consumption per Hour (sorted):", hour_avg.sortByKey().collect())
    
    # Step 5: Group By Key
    group_data = pair_rdd.groupByKey()
    print("Sample GroupByKey first item key & count of values:")
    first_item = group_data.first()
    print(f"Hour: {first_item[0]}, Count: {len(list(first_item[1]))}")
    
    # Step 6: Sort By Key
    sorted_hour = hour_consumption.sortByKey()
    print("Sorted total consumption by hour (top 3):", sorted_hour.take(3))
    
    # Step 7: Shuffle Operation (reduceByKey automatically shuffles)
    print("reduceByKey operation executed (involves shuffling).")
    
    # Step 8: Cache RDD
    pair_rdd.cache()
    print("Is pair RDD cached?", pair_rdd.is_cached)
    
    # Step 9: Persist RDD & Unpersist
    if pair_rdd.is_cached:
        pair_rdd.unpersist()
    pair_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    print("Storage level after persist:", pair_rdd.getStorageLevel())
    
    return pair_rdd

def run_dataframe_operations(df):
    """Q4: Spark DataFrame Operations."""
    print("\n--- Q4: Spark DataFrame Operations ---")
    
    # Step 2 & 3: Selection
    df.select("Temperature", "Humidity", "Zone1").show(5)
    df.select("Datetime", "Temperature", "Humidity", "WindSpeed").show(5)
    
    # Step 4 & 5: Filtering
    print("Filter Temp > 35 count:", df.filter(df.Temperature > 35).count())
    print("Filter Zone1 > 35000 count:", df.filter(df.Zone1 > 35000).count())
    
    # Step 6: Filter multiple conditions
    print("Filter Temp > 30 and Humidity < 50 count:", df.filter((df.Temperature > 30) & (df.Humidity < 50)).count())
    
    # Step 7 & 8: Create new columns
    df_with_diff = df.withColumn("Consumption_Difference", col("Zone1") - col("PowerConsumption_Zone2"))
    df_with_diff.select("Zone1", "PowerConsumption_Zone2", "Consumption_Difference").show(5)
    
    df_all_cols = df.withColumn(
        "Total_Consumption",
        col("Zone1") + col("PowerConsumption_Zone2") + col("PowerConsumption_Zone3")
    )
    df_all_cols.select("Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3", "Total_Consumption").show(5)
    
    # Step 9: Sort by Temperature descending
    df.orderBy(df.Temperature.desc()).show(5)
    
    # Step 10: Group by Humidity and average consumption
    df.groupBy("Humidity").agg(avg("Zone1").alias("Average_Consumption")).show(5)
    
    # Step 15 & 16: Distinct & Count distinct
    print("Distinct Humidity values count:", df.select("Humidity").distinct().count())
    
    # Step 18: Rename Column (Zone1 is already renamed from PowerConsumption_Zone1, print schema)
    df.printSchema()
    
    # Step 19: Split and join
    df_weather = df.select("Datetime", "Temperature", "Humidity", "WindSpeed")
    df_power = df.select("Datetime", "Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3")
    joined_df = df_weather.join(df_power, on="Datetime", how="inner")
    print("Joined DataFrame (first 3 rows):")
    joined_df.show(3)
    
    # Step 20: Aggregation
    df.agg(
        avg("Zone1").alias("Average"),
        _max("Zone1").alias("Maximum"),
        _min("Zone1").alias("Minimum")
    ).show()
    
    return df_all_cols

def run_spark_sql_queries(spark, df):
    """Q5: Exploratory Data Analysis and Spark SQL."""
    print("\n--- Q5: Exploratory Data Analysis & Spark SQL ---")
    
    # Enrich the dataframe with date/time features
    df_enriched = df.withColumn("Hour", hour("Datetime")) \
                    .withColumn("Day", dayofmonth("Datetime")) \
                    .withColumn("Month", month("Datetime")) \
                    .withColumn("Year", year("Datetime")) \
                    .withColumn("MonthName", date_format("Datetime", "MMMM"))
                    
    # Consumer Category Classification
    low_thresh, high_thresh = df_enriched.approxQuantile("Total_Consumption", [0.33, 0.66], 0.01)
    print(f"Low/Medium Threshold: {low_thresh:.2f}, Medium/High Threshold: {high_thresh:.2f}")
    
    df_enriched = df_enriched.withColumn(
        "Consumer_Category",
        when(col("Total_Consumption") <= low_thresh, "Low")
        .when(col("Total_Consumption") <= high_thresh, "Medium")
        .otherwise("High")
    )
    
    # Register temporary SQL view
    df_enriched.createOrReplaceTempView("energy_data")
    
    # 1. Correlation Analysis (Weather vs Consumption)
    print("--- Correlation between Weather and Consumption ---")
    spark.sql("""
        SELECT corr(Temperature, Total_Consumption) AS Temp_Correlation,
               corr(Humidity, Total_Consumption) AS Humidity_Correlation,
               corr(WindSpeed, Total_Consumption) AS WindSpeed_Correlation
        FROM energy_data
    """).show()
    
    # 2. Hourly Energy Consumption Analysis
    print("--- Hourly Average Consumption ---")
    spark.sql("""
        SELECT Hour, ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption, COUNT(*) AS Record_Count
        FROM energy_data
        GROUP BY Hour
        ORDER BY Hour
    """).show(24)
    
    # 3. Region-wise (Zone-wise) Usage
    print("--- Region-wise Consumption Totals and Averages ---")
    spark.sql("""
        SELECT ROUND(SUM(Zone1), 2) AS Zone1_Total,
               ROUND(SUM(PowerConsumption_Zone2), 2) AS Zone2_Total,
               ROUND(SUM(PowerConsumption_Zone3), 2) AS Zone3_Total,
               ROUND(AVG(Zone1), 2) AS Zone1_Avg,
               ROUND(AVG(PowerConsumption_Zone2), 2) AS Zone2_Avg,
               ROUND(AVG(PowerConsumption_Zone3), 2) AS Zone3_Avg
        FROM energy_data
    """).show()
    
    # 4. Identify Peak Demand Periods
    print("--- Top 5 Peak Demand Hours ---")
    spark.sql("""
        SELECT Hour, ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption
        FROM energy_data
        GROUP BY Hour
        ORDER BY Avg_Consumption DESC
        LIMIT 5
    """).show()
    
    # 5. Analyze Consumer Categories
    print("--- Consumer Category Distribution ---")
    spark.sql("""
        SELECT Consumer_Category, COUNT(*) AS Record_Count, ROUND(AVG(Total_Consumption), 2) AS Avg_Consumption
        FROM energy_data
        GROUP BY Consumer_Category
        ORDER BY Avg_Consumption DESC
    """).show()
    
    # 6. Generate Monthly Consumption Reports
    print("--- Monthly Consumption Report ---")
    spark.sql("""
        SELECT MonthName, Month,
               ROUND(SUM(Total_Consumption), 2) AS Total_Monthly_Consumption,
               ROUND(AVG(Total_Consumption), 2) AS Avg_Monthly_Consumption,
               ROUND(MAX(Total_Consumption), 2) AS Peak_Monthly_Consumption
        FROM energy_data
        GROUP BY MonthName, Month
        ORDER BY Month
    """).show(12)
    
    return df_enriched

def run_etl_pipeline(spark, df, output_dir):
    """Q6: ETL Pipeline Development."""
    print("\n--- Q6: ETL Pipeline Development ---")
    
    # EXTRACT step
    raw_count = df.count()
    print(f"ETL Extract: {raw_count} records extracted.")
    
    # TRANSFORM step: Clean and Enrich
    # Remove duplicate records
    cleaned_df = df.dropDuplicates()
    
    # Enrich with columns
    enriched_df = cleaned_df.withColumn("Hour", hour("Datetime")) \
                             .withColumn("Day", dayofmonth("Datetime")) \
                             .withColumn("Month", month("Datetime")) \
                             .withColumn("Year", year("Datetime")) \
                             .withColumn("MonthName", date_format("Datetime", "MMMM")) \
                             .withColumn("Total_Consumption", col("Zone1") + col("PowerConsumption_Zone2") + col("PowerConsumption_Zone3"))
                             
    low_thresh, high_thresh = enriched_df.approxQuantile("Total_Consumption", [0.33, 0.66], 0.01)
    enriched_df = enriched_df.withColumn(
        "Consumer_Category",
        when(col("Total_Consumption") <= low_thresh, "Low")
        .when(col("Total_Consumption") <= high_thresh, "Medium")
        .otherwise("High")
    )
    
    # Data Quality Report
    print(f"ETL Quality Check: Null counts in Datetime = {enriched_df.filter(col('Datetime').isNull()).count()}")
    
    # Split into smart_meter_fact and weather_dim
    smart_meter_fact = enriched_df.select(
        "Datetime", "Day", "Hour", "Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3", "Total_Consumption", "Year", "Month"
    )
    weather_dim = enriched_df.select(
        "Datetime", "Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows"
    )
    
    # LOAD step: Persist to Parquet partitioned by Year and Month
    fact_output = os.path.join(output_dir, "smart_meter_fact")
    weather_output = os.path.join(output_dir, "weather_dim")
    
    smart_meter_fact.write.mode("overwrite").partitionBy("Year", "Month").parquet(fact_output)
    weather_dim.write.mode("overwrite").parquet(weather_output)
    
    print(f"Fact Table loaded to Parquet at: {fact_output}")
    print(f"Weather Dim Table loaded to Parquet at: {weather_output}")
    
    return smart_meter_fact, weather_dim

def run_ml_prediction(df_enriched, images_dir):
    """Q7: Machine Learning/Deep Learning Implementation."""
    print("\n--- Q7: Machine Learning Implementation ---")
    
    # Features assembly
    feature_cols = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows", "Hour", "Month"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    ml_data = assembler.transform(df_enriched).select("Datetime", "features", col("Total_Consumption").alias("label"))
    
    # Train-test split
    train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training count: {train_data.count()}, Test count: {test_data.count()}")
    
    # Models to train
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
        print(f"Model: {name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
    # Find model with lowest RMSE
    best_model_info = min(model_metrics, key=lambda x: x[1])
    best_model_name = best_model_info[0]
    best_model = fitted_models[best_model_name]
    print(f"\nBest Model selected: {best_model_name} with RMSE = {best_model_info[1]:.2f}")
    
    # Anomaly Detection
    best_preds = best_model.transform(test_data)
    best_preds = best_preds.withColumn("abs_residual", _abs(col("label") - col("prediction")))
    
    residual_stats = best_preds.select("abs_residual").summary("mean", "stddev").collect()
    mean_res = float(residual_stats[0]["abs_residual"])
    std_res = float(residual_stats[1]["abs_residual"])
    anomaly_threshold = mean_res + 3 * std_res
    print(f"Residual stats -> Mean: {mean_res:.2f}, StdDev: {std_res:.2f}, Anomaly Threshold: {anomaly_threshold:.2f}")
    
    anomalies = best_preds.withColumn(
        "is_anomaly",
        when(col("abs_residual") > anomaly_threshold, True).otherwise(False)
    )
    
    anomaly_count = anomalies.filter(col("is_anomaly") == True).count()
    print(f"Detected {anomaly_count} anomalies out of {anomalies.count()} records.")
    
    # Generate scatter plot
    os.makedirs(images_dir, exist_ok=True)
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
    plt.title(f"Actual vs Predicted Demand Anomalies - {best_model_name}")
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(images_dir, "demand_anomalies.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved anomalies chart to: {plot_path}")
    
    return best_model

def main():
    spark = init_spark()
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "powerconsumption.csv")
        output_dir = os.path.join(base_dir, "output")
        images_dir = os.path.join(base_dir, "images")
        
        # Q1: Load
        df = load_dataset(spark, file_path)
        
        # Q2: RDD Operations
        rdd = run_rdd_operations(spark, df)
        
        # Q3: Key-Value operations & Persistence
        run_key_value_operations(rdd)
        
        # Q4: DataFrame operations
        df_all = run_dataframe_operations(df)
        
        # Q5: EDA & SQL
        df_enriched = run_spark_sql_queries(spark, df_all)
        
        # Q6: ETL
        run_etl_pipeline(spark, df_enriched, output_dir)
        
        # Q7: ML and anomalies
        run_ml_prediction(df_enriched, images_dir)
        
        print("\nAll Spark analytics tasks completed successfully.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
