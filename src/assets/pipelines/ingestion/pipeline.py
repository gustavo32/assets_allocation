from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_ifix_data,
                inputs=["params:start_date", "params:end_date"],
                outputs="ifix",
                name="ifix_data_node",
            ),
            node(
                func=get_bvsp_data,
                inputs=["params:start_date", "params:end_date"],
                outputs="bvsp",
                name="bvsp_data_node",
            ),
            node(
                func=get_global_vix_data,
                inputs=["params:start_date", "params:end_date"],
                outputs="global_vix",
                name="global_vix_data_node",
            ),
            node(
                func=get_brazil_vix_data,
                inputs=["params:start_date", "params:end_date"],
                outputs="brazil_vix",
                name="brazil_vix_data_node",
            ),
            node(
                func=get_ivvb11_data,
                inputs=["params:start_date", "params:end_date"],
                outputs=["ivvb11", "business_dates"],
                name="ivvb11_data_node",
            ),
            node(
                func=get_ipca_data,
                inputs=["business_dates"],
                outputs="posfixado_ipca",
                name="ipca_data_node",
            ),
            node(
                func=get_cdi_data,
                inputs=["business_dates"],
                outputs="posfixado_cdi",
                name="cdi_data_node",
            ),
        ],
    )
