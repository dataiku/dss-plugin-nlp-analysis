# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_input_names_for_role
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging
from tagger import Tagger

text_input = get_input_names_for_role("document_dataset")[0]
ontology_input = get_input_names_for_role("ontology_dataset")[0]

settings = DkuConfigLoadingOntologyTagging(text_input, ontology_input).load_settings()
text_dataframe = settings.text_input.get_dataframe(infer_with_pandas=False)
ontology_dataframe = settings.ontology_input.get_dataframe(
    columns=settings.ontology_columns, infer_with_pandas=False
)

tagger = Tagger(
    text_df=text_dataframe,
    ontology_df=ontology_dataframe,
    text_column=settings.text_column,
    language=settings.language,
    language_column=settings.language_column,
    tag_column=settings.tag_column,
    category_column=settings.category_column,
    keyword_column=settings.keyword_column,
    lemmatization=settings.lemmatization,
    case_insensitive=settings.case_insensitive,
    normalization=settings.unicode_normalization,
    mode=settings.output_format,
)

output_df = tagger.tag_and_format()
settings.output_dataset.write_with_schema(output_df)