"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from assets.pipelines import preprocessing as dp
from assets.pipelines import training as ds
from assets.pipelines import ingestion as di


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    ingestion_pipeline = di.create_pipeline()
    preprocessing_pipeline = dp.create_pipeline()
    training_pipeline = ds.create_pipeline()

    return {
        "__default__": ingestion_pipeline + preprocessing_pipeline + training_pipeline,
        "science": preprocessing_pipeline + training_pipeline,
        "ingestion": ingestion_pipeline,
        "preprocessing": preprocessing_pipeline,
        "training": training_pipeline,
    }
