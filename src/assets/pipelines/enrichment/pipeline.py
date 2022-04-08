from functools import reduce
from operator import add

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.framework.session import get_current_session

from .nodes import *

def create_enrich_cdi_pipeline():
    return Pipeline(
        [
            node(
                func=cdi_real_projection, # cdi - ipca projections
                inputs=["resampled_ipca", "params:posfixado_ipca_return", "business_dates", "params:resample_rate"],
                outputs="posfixado_ipca_investment",
                name="posfixado_ipca_investment_node"
            ),
        ]
    )

# CDI
# Taxa de juros real Projection

# BVSP
# PIB level against others countries
# Taxa de juros real projection against others
# Fluxo Cambial Estrangeiro
# Produção Industrial
# Rating
# IMF

# IFIX
#



def create_pipeline(**kwargs) -> Pipeline:
    session = get_current_session()
    context = session.load_context()
    catalog = context.catalog
    resample_strategies = catalog.load("params:resample_strategies")
    indices = catalog.load("params:indices")

    resample_pipeline = create_resample_pipeline(resample_strategies)
    investments_pipeline = create_investments_pipeline()
    features_pipeline = create_features_pipeline(indices)

    return resample_pipeline + investments_pipeline + features_pipeline