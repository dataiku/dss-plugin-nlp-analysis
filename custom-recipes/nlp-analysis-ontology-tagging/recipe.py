# -*- coding: utf-8 -*-
import dataiku
from formatter_instanciator import COLUMNS_DESCRIPTION
from dkulib_io_utils import set_column_descriptions
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
    case_insensitivity=settings.case_insensitivity,
    normalization=settings.unicode_normalization,
)
output_df = tagger.tag_and_format(
    text_df=text_dataframe,
    text_column=settings.text_column,
    language_column=settings.language_column,
    output_format=settings.output_format,
    languages=languages,
)
settings.output_dataset.write_with_schema(output_df)
set_column_descriptions(
    output_dataset=settings.output_dataset, column_descriptions=COLUMNS_DESCRIPTION
)
