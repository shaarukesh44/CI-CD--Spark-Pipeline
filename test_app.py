import pytest
from app import create_spark_session, run_pipeline


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session()
    yield session
    session.stop()


def test_spark_pipeline(spark):

    predictions, accuracy = run_pipeline(spark)

    assert predictions.count() == 10

    assert "prediction" in predictions.columns

    assert 0 <= accuracy <= 1


def test_model_accuracy(spark):

    predictions, accuracy = run_pipeline(spark)

    # Model validation threshold
    assert accuracy >= 0.80