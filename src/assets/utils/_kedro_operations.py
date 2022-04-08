def get_output_node_names(pipeline_name, endswith):
    return [k.replace(endswith, "") for k in pipeline_name._nodes_by_output.keys() if
     k.endswith(endswith)]