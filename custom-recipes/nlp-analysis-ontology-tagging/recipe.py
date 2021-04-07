# -*- coding: utf-8 -*-
import dataiku
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging
from ontology_tagger import Tagger


dku_config = DkuConfigLoadingOntologyTagging()
settings = dku_config.load_settings()
text_dataframe = settings.text_input.get_dataframe(infer_with_pandas=False)
ontology_dataframe = settings.ontology_input.get_dataframe(
    columns=settings.ontology_columns, infer_with_pandas=False
)

languages = (
            text_dataframe[settings.language_column].dropna().unique()
            if settings.language == "language_column"
            else [settings.language]
        )
dku_config._check_languages(languages)

tagger = Tagger(
    ontology_df=ontology_dataframe,
    tag_column=settings.tag_column,
    category_column=settings.category_column,
    keyword_column=settings.keyword_column,
    language=settings.language,
    lemmatization=settings.lemmatization,
    case_insensitive=settings.case_insensitive,
    normalization=settings.unicode_normalization,
)

output_df = tagger.tag_and_format(
    text_dataframe,
    settings.text_column,
    settings.language_column,
    settings.output_format,
    languages
)
settings.output_dataset.write_with_schema(output_df)