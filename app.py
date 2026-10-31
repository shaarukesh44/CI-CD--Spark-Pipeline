from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def create_spark_session():
    return SparkSession.builder \
        .appName("Spark CI CD Application") \
        .master("local[2]") \
        .getOrCreate()


def run_pipeline(spark):

    # Create sample dataset
    data = [
        (20.0, 20000.0, 0.0),
        (25.0, 25000.0, 0.0),
        (30.0, 30000.0, 0.0),
        (35.0, 60000.0, 1.0),
        (40.0, 70000.0, 1.0),
        (45.0, 80000.0, 1.0),
        (50.0, 90000.0, 1.0),
        (22.0, 22000.0, 0.0),
        (38.0, 65000.0, 1.0),
        (28.0, 28000.0, 0.0)
    ]

    columns = ["age", "salary", "label"]

    df = spark.createDataFrame(data, columns)

    # Feature engineering
    assembler = VectorAssembler(
        inputCols=["age", "salary"],
        outputCol="features"
    )

    # Machine learning model
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label"
    )

    pipeline = Pipeline(stages=[assembler, lr])

    # Train model
    model = pipeline.fit(df)

    # Generate predictions
    predictions = model.transform(df)

    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)

    return predictions, accuracy


if __name__ == "__main__":

    spark = create_spark_session()

    predictions, accuracy = run_pipeline(spark)

    predictions.show()

    print("Model Accuracy:", accuracy)

    spark.stop()