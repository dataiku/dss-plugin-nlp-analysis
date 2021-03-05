# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_output_names_for_role
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging, DkuConfigLoading
from tagger import Tagger

"""Give settings to the Tagger and get the created output dataset"""


def call_tagger(settings):
    tagging = Tagger(settings)
    df = tagging.tagging_proceedure()
    return df


"""Main public function"""


def process_params():

    settings = DkuConfigLoadingOntologyTagging().load_settings()
    tagged_documents_df = call_tagger(settings)
    output_ds = get_output_names_for_role("tagged_documents")[0]
    output_ds = dataiku.Dataset(output_ds)
    output_ds.write_with_schema(tagged_documents_df)


# Run
process_params()
