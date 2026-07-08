import pytest
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from src.spark_analytics import (
    init_spark,
    load_dataset,
    run_rdd_operations,
    run_key_value_operations,
    run_dataframe_operations,
    run_etl_pipeline
)

@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder \
        .appName("Test Smart Energy Consumption") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    yield spark
    spark.stop()

def test_load_dataset(spark_session):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "powerconsumption.csv")
    df = load_dataset(spark_session, file_path)
    
    assert df.count() > 0
    assert "Datetime" in df.columns
    assert "Temperature" in df.columns
    assert "Humidity" in df.columns
    assert "Zone1" in df.columns

def test_rdd_operations(spark_session):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "powerconsumption.csv")
    df = load_dataset(spark_session, file_path)
    
    # We slice a small subset to run RDD tests quickly
    df_sample = df.limit(100)
    rdd = run_rdd_operations(spark_session, df_sample)
    
    assert rdd.count() == 100

def test_key_value_operations(spark_session):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "powerconsumption.csv")
    df = load_dataset(spark_session, file_path)
    df_sample = df.limit(50)
    rdd = df_sample.rdd
    
    pair_rdd = run_key_value_operations(rdd)
    assert pair_rdd.count() == 50

def test_dataframe_operations(spark_session):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "powerconsumption.csv")
    df = load_dataset(spark_session, file_path)
    df_sample = df.limit(100)
    
    df_all = run_dataframe_operations(df_sample)
    assert "Consumption_Difference" in df_all.columns
    assert "Total_Consumption" in df_all.columns

def test_etl_pipeline(spark_session, tmpdir):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "powerconsumption.csv")
    df = load_dataset(spark_session, file_path)
    df_sample = df.limit(100)
    
    output_dir = str(tmpdir.mkdir("output"))
    smart_meter_fact, weather_dim = run_etl_pipeline(spark_session, df_sample, output_dir)
    
    # Check that Parquet files were created
    assert os.path.exists(os.path.join(output_dir, "smart_meter_fact"))
    assert os.path.exists(os.path.join(output_dir, "weather_dim"))
    assert smart_meter_fact.count() == 100
    assert weather_dim.count() == 100
