from fastapi import FastAPI, Query
import uvicorn
import os
import sys

# Ensure parent directory is in PYTHONPATH so we can import src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spark_analytics import init_spark, load_dataset, run_dataframe_operations, run_spark_sql_queries, run_ml_prediction

app = FastAPI(
    title="Smart Energy Consumption Analytics API",
    description="REST API for predicting electricity demand and detecting anomalies using Apache Spark MLlib.",
    version="1.0.0"
)

# Global variables to cache model and spark session if possible, 
# but we can re-train/run dynamically as requested.
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Smart Energy Consumption Analytics Platform API",
        "status": "running",
        "endpoints": {
            "status": "GET /",
            "predict": "GET /predict?Temperature=30&Humidity=40&WindSpeed=1.5&GeneralDiffuseFlows=500&DiffuseFlows=100&Hour=15&Month=7"
        }
    }

@app.get("/predict")
def predict_demand(
    Temperature: float = Query(..., description="Temperature in °C"),
    Humidity: float = Query(..., description="Humidity percentage (0-100)"),
    WindSpeed: float = Query(..., description="Wind speed in m/s"),
    GeneralDiffuseFlows: float = Query(..., description="General Diffuse Flows"),
    DiffuseFlows: float = Query(..., description="Diffuse Flows"),
    Hour: int = Query(..., description="Hour of the day (0-23)"),
    Month: int = Query(..., description="Month of the year (1-12)")
):
    spark = init_spark()
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "powerconsumption.csv")
        images_dir = os.path.join(base_dir, "images")
        
        # Load and process data to train model
        df = load_dataset(spark, file_path)
        df_all = run_dataframe_operations(df)
        df_enriched = run_spark_sql_queries(spark, df_all)
        
        # Train and get the best model
        model = run_ml_prediction(df_enriched, images_dir)
        
        if model is None:
            return {"error": "Could not train prediction model."}
            
        # Create input features DataFrame for prediction
        input_data = spark.createDataFrame(
            [(Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows, Hour, Month)],
            ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows", "Hour", "Month"]
        )
        
        from pyspark.ml.feature import VectorAssembler
        feature_cols = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows", "Hour", "Month"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        feature_df = assembler.transform(input_data)
        
        predictions = model.transform(feature_df)
        result = predictions.select("prediction").first()
        predicted_val = float(result.prediction)
        
        return {
            "features": {
                "Temperature": Temperature,
                "Humidity": Humidity,
                "WindSpeed": WindSpeed,
                "GeneralDiffuseFlows": GeneralDiffuseFlows,
                "DiffuseFlows": DiffuseFlows,
                "Hour": Hour,
                "Month": Month
            },
            "prediction": {
                "predicted_demand": round(predicted_val, 2),
                "unit": "kW"
            }
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        print("Stopping Spark session inside API request...")
        spark.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
