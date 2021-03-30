# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging
from tagger import Tagger

text_input = get_input_names_for_role("document_dataset")[0]
ontology_input = get_input_names_for_role("ontology_dataset")[0]

settings = DkuConfigLoadingOntologyTagging(text_input, ontology_input).load_settings()

tagger = Tagger(
    settings.text_input.get_dataframe(),
    settings.ontology_input.get_dataframe(),
    settings.text_column,
    settings.language,
    settings.language_column,
    settings.tag_column,
    settings.category_column,
    settings.keyword_column,
    settings.lemmatization,
    settings.case_insensitive,
    settings.unicode_normalization,
    settings.output_format,
)
output_df = tagger.tag_and_format()

output_dataset = get_output_names_for_role("tagged_documents")[0]
output_dataset = dataiku.Dataset(output_dataset)
output_dataset.write_with_schema(output_df)