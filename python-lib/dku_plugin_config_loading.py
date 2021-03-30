# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_recipe_config, get_output_names_for_role
from language_dict import SUPPORTED_LANGUAGES_SPACY
from dku_config import DkuConfig


class DkuConfigLoading:
    def __init__(self):
        self.config = get_recipe_config()
        self.dku_config = DkuConfig()


class DkuConfigLoadingOntologyTagging(DkuConfigLoading):
    """Configuration for Ontology Tagging Plugin"""

    MATCHING_PARAMETERS = [
        "case_insensitive",
        "lemmatization",
        "unicode_normalization",
    ]

    def __init__(self, text_input, ontology_input):
        """Instanciate class with DkuConfigLoading and add input datasets to dku_config"""

        super().__init__()
        self.dku_config.add_param(
            name="text_input", value=dataiku.Dataset(text_input), required=True
        )
        self.dku_config.add_param(
            name="ontology_input", value=dataiku.Dataset(ontology_input), required=True
        )
        self.document_dataset_columns = [
            p["name"] for p in self.dku_config.text_input.read_schema()
        ]
        self.ontology_dataset_columns = [
            p["name"] for p in self.dku_config.ontology_input.read_schema()
        ]

    def _content_error_message(self, error, column):
        """Get corresponding error message if any"""

        if error == "missing":
            return "Missing input column."

        if error == "invalid":
            return "Invalid input column : {}.\n".format(column)

        if error == "language":
            return "You must select one of the languages.\n If your dataset contains several languages, you can use 'Language column' and create a column in your Document dataset containing the languages of the documents.\n"

    def _add_text_column(self):
        """Load text column from Document Dataset"""

        text_column = self.config.get("text_column")
        input_columns = self.document_dataset_columns
        self.dku_config.add_param(
            name="text_column",
            value=text_column,
            required=True,
            checks=self._get_column_checks(text_column, input_columns),
        )

    def _add_matching_settings(self):
        """Load matching parameters"""

        for parameter in self.MATCHING_PARAMETERS:
            self.dku_config.add_param(
                name=parameter, value=self.config[parameter], required=True
            )

    def _add_language(self):
        """Load language from dropdown"""

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

    def _add_language_column(self):
        """Load language column if specified"""
        lang_column = self.config.get("language_column")
        input_columns = self.document_dataset_columns
        self.dku_config.add_param(
            name="language_column",
            value=lang_column,
            checks=[
                {
                    "type": "custom",
                    "cond": bool(lang_column),
                    "err_msg": self._content_error_message("missing", None),
                },
                {
                    "type": "custom",
                    "cond": lang_column in input_columns + [None]
                    or not bool(lang_column),
                    "err_msg": self._content_error_message("invalid", lang_column),
                },
            ],
        )

    def _get_column_checks(self, column, input_columns):
        """Check for mandatory columns parameters"""

        return [
            {
                "type": "exists",
                "err_msg": self._content_error_message("missing", None),
            },
            {
                "type": "custom",
                "cond": column in input_columns or column == None,
                "err_msg": self._content_error_message("invalid", column),
            },
        ]

    def _ontology_columns_mandatory(self, column_name, column_label, input_columns):
        """Load mandatory columns from Ontology Dataset"""

        column = self.config.get(column_name)
        self.dku_config.add_param(
            name=column_name,
            value=column,
            required=True,
            checks=self._get_column_checks(column, input_columns),
        )

    def _add_ontology_columns(self):
        """Load columns from Ontology Dataset"""

        input_columns = self.ontology_dataset_columns
        self._ontology_columns_mandatory("tag_column", "Tag column", input_columns)
        self._ontology_columns_mandatory(
            "keyword_column", "Keyword column", input_columns
        )
        ontology_columns = [self.dku_config.tag_column,self.dku_config.keyword_column]
        self._add_category_column()
        category_column = self.dku_config.category_column
        if category_column : 
            ontology_columns.append(category_column)
        self.dku_config.add_param(
        name="ontology_columns",
        value=ontology_columns)

    def _add_category_column(self):
        """Load category column if exists"""

        category_column = self.config.get("category_column")
        input_columns = self.ontology_dataset_columns
        self.dku_config.add_param(
            name="category_column",
            value=category_column if category_column else None,
            required=False,
            checks=[
                {
                    "type": "in",
                    "op": input_columns + [None],
                    "err_msg": self._content_error_message("invalid", category_column),
                }
            ],
        )

    def _add_output_format(self):
        """Load output format parameters"""

        output = self.config.get("output_format")

        self.dku_config.add_param(
            name="output_format",
            value=output,
            required=True,
        )
    
    def _add_output_dataset(self):
        output_dataset_name = get_output_names_for_role("tagged_documents")[0]
        self.dku_config.add_param(
            name="output_dataset", value=dataiku.Dataset(output_dataset_name), required=True
        )

    def load_settings(self):
        """Public function to load all given parameters for Ontology Tagging Plugin"""

        self._add_matching_settings()
        self._add_text_column()
        self._add_language()
        
        if self.dku_config.language == "language_column":
            self._add_language_column()
        else:
            self.dku_config.add_param(
                name="language_column",
                value="",
            )
            
        self._add_ontology_columns()
        self._add_output_format()
        self._add_output_dataset()
        return self.dku_config