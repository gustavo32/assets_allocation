from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.framework.session import get_current_session

from .nodes import *


def create_pre_modeling_pipeline():
    return Pipeline(
        [
            node(
                func=separate_labels_and_features,
                inputs=["model_input"],
                outputs=["X", "y"],
                name="separate_labels_and_features_node",
            ),
            node(
                func=split_data,
                inputs=["X", "y", "params:training.training_size"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )


def create_modeling_pipeline(experiments):
    def _pipeline(name):
        return pipeline([
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:indices", "params:experiment"],
                outputs="models",
                name=name+"_train_node",
            ),
            node(
                func=evaluate_model,
                inputs=["models", "X_test", "y_test", "params:indices", "params:experiment"],
                outputs="y_preds",
                name=name+"_evaluate_node",
            ),
            node(
                func=compare_performance_traditional_scenario,
                inputs=["X", "y", "y_preds", "params:make_plots"],
                outputs=None,
                name=name+"_compare_performance_node",
            ),
        ])

    list_modeling_pipeline = [
        pipeline(
            pipe=_pipeline(name=experiment),
            parameters={
                "params:experiment": f"params:training.experiments.{experiment}",
                "params:make_plots": "params:training.make_plots",
            },
            # inputs={"df": experiment+"_experiment"},
            outputs={
                "models": f"{experiment}_models",
                "y_preds": f"{experiment}_preds",
            },
        )
        for experiment in experiments
    ]

    # Namespace consolidated modeling pipelines
    return pipeline(pipe=reduce(add, list_modeling_pipeline))


def create_merge_pipeline(inputs, endswith):
    inputs = {k: k+endswith for k in inputs}
    inputs["delete_first_n_rows"] = "params:delete_first_n_rows"
    return Pipeline(
        [
            node(
                func=merge_feat_tables,
                inputs=inputs,
                outputs="model_input",
                name="model_input"
            )
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    session = get_current_session()
    context = session.load_context()
    catalog = context.catalog
    experiments = catalog.load("params:training.experiments")
    indices = catalog.load("params:indices")

    merge_pipeline = create_merge_pipeline(indices.keys(), "_features")
    pre_modeling_pipeline = create_pre_modeling_pipeline()
    modeling_pipeline = create_modeling_pipeline(experiments)

    return merge_pipeline + pre_modeling_pipeline + modeling_pipeline

    # return pipeline(
    #     pipe=ds_pipeline_1 + ds_pipeline_2,
    #     inputs="model_input_table",
    #     namespace="training",
    # )
