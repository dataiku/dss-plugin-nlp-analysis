# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging, DkuConfigLoading
from tagger import Tagger

"""Return input dataframe and the list of its columns"""


def get_input_dataframe(input_dataset):
    input_name = get_input_names_for_role(input_dataset)[0]
    dataframe = dataiku.Dataset(input_name).get_dataframe()
    dataframe_columns = dataframe.columns.tolist()
    return dataframe, dataframe_columns


"""Give settings to the Tagger and get the created output dataset"""


def call_tagger(settings, text_input, ontology_input):
    tagging = Tagger(settings, text_input, ontology_input)
    df = tagging.tagging_proceedure()
    return df


"""Main public function"""


def process_params():
    text_input, text_input_columns = get_input_dataframe("document_dataset")
    ontology_input, ontology_input_columns = get_input_dataframe("ontology_dataset")

    settings = DkuConfigLoadingOntologyTagging(
        text_input_columns, ontology_input_columns
    ).load_settings()

    tagged_documents_df = call_tagger(settings, text_input, ontology_input)

    output_ds = get_output_names_for_role("tagged_documents")[0]
    output_ds = dataiku.Dataset(output_ds)
    output_ds.write_with_schema(tagged_documents_df)


# Run
process_params()
