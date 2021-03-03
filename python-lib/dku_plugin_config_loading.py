# -*- coding: utf-8 -*-
from dataiku.customrecipe import *
import dataiku
import pandas as pd
from dataiku import pandasutils as pdu
from dataiku.customrecipe import get_recipe_config
from language_dict import SUPPORTED_LANGUAGES_SPACY
from dku_config import DkuConfig

# All plugin parameters name
MATCHING_PARAMS = ["case_sensitivity", "lemmatization", "unicode_normalization"]


class DkuConfigLoading:
    def __init__(self):

        self.config = get_recipe_config()
        self.dku_config = DkuConfig()
        self.add_input_ds()

    def content_err_msg(self, err, par):
        if err == "missing":
            return "Missing input column : {}.\n".format(par)

        if err == "invalid":
            return "Invalid input column : {}.\n".format(par)

        if err == "language":
            return "You must select one of the languages.\n If your dataset contains several languages, you can use 'Language column' and create a column in your Document dataset containing the languages of the documents.\n"

    def get_input_datasets(self, role):
        return dataiku.Dataset(get_input_names_for_role(role)[0])

    def add_input_ds(self):
        text_input = self.get_input_datasets("document_dataset").get_dataframe()
        ontology_input = self.get_input_datasets("ontology_dataset").get_dataframe()

        self.input_text_cols = text_input.columns.tolist()
        self.input_onto_cols = ontology_input.columns.tolist()

        self.dku_config.add_param(name="text_input", value=text_input, required=True)
        self.dku_config.add_param(
            name="ontology_input", value=ontology_input, required=True
        )

    # Load text column from Document Dataset
    def add_text_col(self):
        text_col = self.config.get("text_column")
        ds = self.input_text_cols
        self.dku_config.add_param(
            name="text_column",
            value=text_col,
            required=True,
            checks=self.get_col_checks(text_col, "Text column", ds),
        )

    # Load matching parameters
    def add_matching_settings(self):
        for par in MATCHING_PARAMS:
            self.dku_config.add_param(name=par, value=self.config[par], required=True)

    # load language from dropdown
    def add_lang(self):

        self.dku_config.add_param(
            name="language",
            value=self.config.get("language"),
            required=True,
            checks=[
                {"type": "exists", "err_msg": self.content_err_msg("language", None)},
                {
                    "type": "in",
                    "op": list(SUPPORTED_LANGUAGES_SPACY.keys()) + ["language_column"],
                    "err_msg": self.content_err_msg("language", None),
                },
            ],
        )

    # load language column if specified
    def add_lang_col(self):

        lang_col = self.config.get("language_column")
        ds = self.input_text_cols
        self.dku_config.add_param(
            name="language_column",
            value=lang_col if lang_col else None,
            required=(self.dku_config.language == "language_column"),
            checks=[
                {
                    "type": "custom",
                    "cond": lang_col != "none",
                    "err_msg": self.content_err_msg("missing", "Language column"),
                },
                {
                    "type": "in",
                    "op": ds + [None],
                    "err_msg": self.content_err_msg("invalid", "Language column"),
                },
            ],
        )

    # check for mandatory columns parameters
    def get_col_checks(self, name, label, ds):
        return [
            {"type": "exists", "err_msg": self.content_err_msg("missing", label)},
            {
                "type": "custom",
                "cond": name in ds,
                "err_msg": self.content_err_msg("invalid", label),
            },
        ]

    # load mandatory columns from Ontology Dataset
    def ontology_cols_param(self, col_name, ds=None, col_label=""):
        col = self.config.get(col_name)
        self.dku_config.add_param(
            name=col_name,
            value=col,
            required=True,
            checks=self.get_col_checks(col, col_name, ds),
        )

    # load columns from Ontology Dataset
    def add_ontology_cols(self):
        ds = self.input_onto_cols
        self.ontology_cols_param("tag_column", ds, "Tag column")
        self.ontology_cols_param("keyword_column", ds, "Keyword column")
        self.add_cat_col()

    # load category column if exists
    def add_cat_col(self):
        cat_col = self.config.get("category_column")
        ds = self.input_onto_cols
        self.dku_config.add_param(
            name="category_column",
            value=cat_col if cat_col else "none",
            required=False,
            checks=[
                {
                    "type": "in",
                    "op": ds + ["none"],
                    "err_msg": self.content_err_msg("invalid", "Category column"),
                }
            ],
        )

    # load output format parameters
    def add_output(self):
        output = self.config.get("output_format")

        self.dku_config.add_param(
            name="output_format",
            value=output,
            required=self.dku_config.category_column == None,
        )

        output = self.config.get("output_format_with_categories")
        self.dku_config.add_param(
            name="output_format_with_categories",
            value=output,
            required=self.dku_config.category_column != None,
        )

    # main
    def load_settings(self):

        self.add_matching_settings()
        self.add_text_col()
        self.add_lang()
        self.add_lang_col()
        self.add_ontology_cols()
        self.add_output()
