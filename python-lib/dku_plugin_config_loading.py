# -*- coding: utf-8 -*-
from dataiku.customrecipe import get_input_names_for_role
import pandas as pd
import dataiku
from dataiku.customrecipe import get_recipe_config
from language_dict import SUPPORTED_LANGUAGES_SPACY
from dku_config import DkuConfig


class DkuConfigLoading:
    def __init__(self):
        self.config = get_recipe_config()
        self.dku_config = DkuConfig()


"""Configuration for Ontology Tagging Plugin"""


class DkuConfigLoadingOntologyTagging(DkuConfigLoading):

    MATCHING_PARAMETERS = [
        "case_sensitivity",
        "lemmatization",
        "unicode_normalization",
    ]

    """instanciate class with DkuConfigLoading and add input datasets to dku_config"""

    def __init__(self, text_columns, ontology_columns):
        super().__init__()
        self.text_columns = text_columns
        self.ontology_columns = ontology_columns

    """get corresponding error message if any"""

    def _content_error_message(self, error, column):
        if error == "missing":
            return "Missing input column : {}.\n".format(column)

        if error == "invalid":
            return "Invalid input column : {}.\n".format(column)

        if error == "language":
            return "You must select one of the languages.\n If your dataset contains several languages, you can use 'Language column' and create a column in your Document dataset containing the languages of the documents.\n"

    """Load text column from Document Dataset"""

    def _add_text_column(self):
        text_column = self.config.get("text_column")
        input_columns = self.text_columns
        self.dku_config.add_param(
            name="text_column",
            value=text_column,
            required=True,
            checks=self._get_column_checks(text_column, "Text column", input_columns),
        )

    """Load matching parameters"""

    def _add_matching_settings(self):
        for parameter in self.MATCHING_PARAMETERS:
            self.dku_config.add_param(
                name=parameter, value=self.config[parameter], required=True
            )

    """Load language from dropdown"""

    def _add_language(self):

        self.dku_config.add_param(
            name="language",
            value=self.config.get("language"),
            required=True,
            checks=[
                {
                    "type": "exists",
                    "err_msg": self._content_error_message("language", None),
                },
                {
                    "type": "in",
                    "op": list(SUPPORTED_LANGUAGES_SPACY.keys()) + ["language_column"],
                    "err_msg": self._content_error_message("language", None),
                },
            ],
        )

    """Load language column if specified"""

    def _add_language_column(self):

        lang_column = self.config.get("language_column")
        input_columns = self.text_columns
        self.dku_config.add_param(
            name="language_column",
            value=lang_column if lang_column else None,
            required=(self.dku_config.language == "language_column"),
            checks=[
                {
                    "type": "custom",
                    "cond": lang_column != None,
                    "err_msg": self._content_error_message(
                        "missing", "Language column"
                    ),
                },
                {
                    "type": "in",
                    "op": input_columns + [None],
                    "err_msg": self._content_error_message(
                        "invalid", "Language column"
                    ),
                },
            ],
        )

    """Check for mandatory columns parameters"""

    def _get_column_checks(self, column, column_name, input_columns):
        return [
            {
                "type": "exists",
                "err_msg": self._content_error_message("missing", column_name),
            },
            {
                "type": "custom",
                "cond": column in input_columns,
                "err_msg": self._content_error_message("invalid", column_name),
            },
        ]

    """Load mandatory columns from Ontology Dataset"""

    def _ontology_columns_mandatory(self, column_name, column_label, input_columns):
        column = self.config.get(column_name)
        self.dku_config.add_param(
            name=column_name,
            value=column,
            required=True,
            checks=self._get_column_checks(column, column_label, input_columns),
        )

    """Load columns from Ontology Dataset"""

    def _add_ontology_columns(self):
        input_columns = self.ontology_columns
        self._ontology_columns_mandatory("tag_column", "Tag column", input_columns)
        self._ontology_columns_mandatory(
            "keyword_column", "Keyword column", input_columns
        )
        self._add_category_column()

    """load category column if exists"""

    def _add_category_column(self):
        category_column = self.config.get("category_column")
        input_columns = self.ontology_columns
        self.dku_config.add_param(
            name="category_column",
            value=category_column if category_column else "none",
            required=False,
            checks=[
                {
                    "type": "in",
                    "op": input_columns + ["none"],
                    "err_msg": self._content_error_message(
                        "invalid", "Category column"
                    ),
                }
            ],
        )

    """load output format parameters"""

    def _add_output(self):
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

    """Public function to load all given parameters for Ontology Tagging Plugin"""

    def load_settings(self):

        self._add_matching_settings()
        self._add_text_column()
        self._add_language()
        self._add_language_column()
        self._add_ontology_columns()
        self._add_output()
        return self.dku_config
