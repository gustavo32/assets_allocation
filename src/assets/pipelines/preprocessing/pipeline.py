from functools import reduce
from operator import add

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.framework.session import get_current_session

from .nodes import *


def create_resample_pipeline(indices):
    def _pipeline(name):
        return Pipeline(
            [
                node(
                    func=resample_data,
                    inputs=["index", "params:resample_rate", "params:resample_strategy"],
                    outputs="resampled_data",
                    name=name + "_resampled"
                ),
            ]
        )


    list_resample_pipeline = [
        pipeline(
            pipe=_pipeline(name=index),
            parameters={
                "params:resample_strategy": f"params:indices.{index}.resample_strategy",
            },
            inputs={"index": index},
            outputs={
                "resampled_data": f"resampled_{index}"
            },
        )
        for index in indices.keys()
    ]

    # Namespace consolidated modeling pipelines
    return pipeline(pipe=reduce(add, list_resample_pipeline))


def create_features_pipeline(indices):
    def _pipeline(name):
        return Pipeline(
            [
                node(
                    func=create_tech_indicators,
                    inputs=["df",
                            "params:tech_indicators",
                            "params:return_risk",
                            "params:add_return_risk"],
                    outputs="features",
                    name=name+"_features_node"
                ),
            ]
        )

    list_features_pipeline = [
        pipeline(
            pipe=_pipeline(name=index),
            parameters={
                "params:tech_indicators": f"params:indices.{index}.process",
                "params:add_return_risk": f"params:indices.{index}.add_return_risk",
            },
            inputs={"df": index+"_investment"},
            outputs={
                "features": f"{index}_features"
            },
        )
        for index in indices
    ]

    # Namespace consolidated modeling pipelines
    return pipeline(pipe=reduce(add, list_features_pipeline))


def create_investments_pipeline():
    return Pipeline(
        [
            node(
                func=prefixado_signal,
                inputs=["resampled_posfixado_ipca", "params:prefixado_return", "business_dates", "params:resample_rate"],
                outputs="prefixado_investment",
                name="prefixado_investment_node"
            ),
            node(
                func=posfixado_cdi_signal,
                inputs=["resampled_posfixado_cdi", "resampled_posfixado_ipca", "params:posfixado_cdi_percentage"],
                outputs="posfixado_cdi_investment",
                name="posfixado_cdi_investment_node"
            ),
            node(
                func=posfixado_ipca_signal,
                inputs=["resampled_posfixado_ipca", "params:posfixado_ipca_return", "business_dates", "params:resample_rate"],
                outputs="posfixado_ipca_investment",
                name="posfixado_ipca_investment_node"
            ),
            node(
                func=bvsp_signal,
                inputs=["resampled_bvsp", "resampled_posfixado_ipca"],
                outputs="bvsp_investment",
                name="bvsp_investment_node"
            ),
            node(
                func=global_vix_signal,
                inputs=["resampled_global_vix", "resampled_posfixado_ipca"],
                outputs="global_vix_investment",
                name="global_vix_investment_node"
            ),
            node(
                func=brazil_vix_signal,
                inputs=["resampled_brazil_vix", "resampled_posfixado_ipca"],
                outputs="brazil_vix_investment",
                name="brazil_vix_investment_node"
            ),
            node(
                func=ifix_signal,
                inputs=["resampled_ifix", "resampled_posfixado_ipca"],
                outputs="ifix_investment",
                name="ifix_investment_node"
            ),
            node(
                func=ivvb11_signal,
                inputs=["resampled_ivvb11", "resampled_posfixado_ipca"],
                outputs="ivvb11_investment",
                name="ivvb11_investment_node"
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    session = get_current_session()
    context = session.load_context()
    catalog = context.catalog
    indices = catalog.load("params:indices")

    resample_pipeline = create_resample_pipeline(indices)
    investments_pipeline = create_investments_pipeline()
    features_pipeline = create_features_pipeline(indices)

    return resample_pipeline + investments_pipeline + features_pipeline