# -*- coding: utf-8 -*-
import dataiku
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging
from tagger import Tagger


settings = DkuConfigLoadingOntologyTagging().load_settings()
text_dataframe = settings.text_input.get_dataframe(infer_with_pandas=False)
ontology_dataframe = settings.ontology_input.get_dataframe(
    columns=settings.ontology_columns, infer_with_pandas=False
)

tagger = Tagger(
    ontology_df=ontology_dataframe,
    tag_column=settings.tag_column,
    category_column=settings.category_column,
    keyword_column=settings.keyword_column,
    language=settings.language,
    lemmatization=settings.lemmatization,
    case_insensitive=settings.case_insensitive,
    normalization=settings.unicode_normalization,
    output_format=settings.output_format,
)

output_df = tagger.tag_and_format(
    text_dataframe, settings.text_column, settings.language_column
)
settings.output_dataset.write_with_schema(output_df)