"""Main module to tag the documents"""

from spacy_tokenizer import MultilingualTokenizer
from dkulib_io_utils import generate_unique
from spacy.matcher import PhraseMatcher
from fastcore.utils import store_attr
from typing import AnyStr, List
import pandas as pd
from tagger_formatter import (
    FormatterByDocumentJson,
    FormatterByDocument,
    FormatterByTag,
)
from enum import Enum
from time import perf_counter
import logging


class OutputFormat(Enum):
    ONE_ROW_PER_TAG = "one_row_per_tag"
    ONE_ROW_PER_DOCUMENT = "one_row_per_doc"
    ONE_ROW_PER_DOCUMENT_JSON = "one_row_per_doc_json"


class Tagger:
    def __init__(
        self,
        text_df,
        ontology_df,
        text_column,
        language,
        language_column,
        tag_column,
        category_column,
        keyword_column,
        lemmatize,
        case_insensitive,
        normalize,
        mode,
    ):
        store_attr()
        self.matcher_dict = {}
        self.nlp_dict = {}

    def _get_patterns(self) -> None:
        """
        Public function called in tagger.py
        Creates the list of patterns :
        - If there aren't category -> list of the keywords (string list)
        - If there are categories  -> list of dictionaries, {"label": category, "pattern": keyword}
        """
        #remove rows with missing values 
        self.ontology_df.replace("", float("nan"), inplace=True) 
        self.ontology_df.dropna(inplace=True)
        assert not(self.ontology_df.empty),"No valid tags were found. Please specify at least a keyword and a tag in the ontology dataset, and restart the plugin"
        list_of_tags = self.ontology_df[self.tag_column].values.tolist()
        list_of_keywords = self.ontology_df[self.keyword_column].values.tolist()
        self.keyword_to_tag = dict(zip(list_of_keywords, list_of_tags))
        
        if self.category_column:
            list_of_categories = self.ontology_df[self.category_column].values.tolist()
            self.patterns = [
                {"label": label, "pattern": pattern}
                for label, pattern in zip(list_of_categories, list_of_keywords)
            ]
        else:
            self.patterns = list_of_keywords

    def _list_sentences(self, row: pd.Series) -> List[AnyStr]:
        """Auxiliary function called in _matching_pipeline
        Applies sentencizer and return list of sentences"""
        document = row[self.text_column]
        if type(document) != str:
            return []
        else:
            return [
                sentence.text
                for sentence in self.nlp_dict[self.language](document).sents
            ]

    def _match_no_category(self, language: AnyStr) -> None:
        """instanciates PhraseMatcher with associated tags"""
        self.nlp_dict[language].remove_pipe("sentencizer")
        matcher = PhraseMatcher(self.nlp_dict[language].vocab)
        self.patterns = list(self.nlp_dict[language].tokenizer.pipe(self.patterns))
        matcher.add("PatternList", self.patterns)
        self.matcher_dict[language] = matcher

    def _match_with_category(self, language: AnyStr) -> None:
        """Instanciates EntityRuler with associated tags and categories"""
        self.nlp_dict[language].remove_pipe("sentencizer")
        ruler = self.nlp_dict[language].add_pipe("entity_ruler")
        ruler.add_patterns(self.patterns)

    def _format_with_category(self) -> pd.DataFrame:
        self._match_with_category(self.language)
        if self.mode == OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value:
            formatter = self.instanciate_class(FormatterByDocumentJson)
            formatter.tag_columns = ["tag_json_full", "tag_json_categories"]

        if self.mode == OutputFormat.ONE_ROW_PER_DOCUMENT.value:
            formatter = self.instanciate_class(FormatterByDocument)
            formatter.tag_columns = ["tag_keywords", "tag_sentences"]

        if self.mode == OutputFormat.ONE_ROW_PER_TAG.value:
            formatter = self.instanciate_class(FormatterByTag)
            formatter.tag_columns = [
                "tag_keyword",
                "tag_sentence",
                "tag_category",
                "tag",
            ]
            return formatter.write_df()
        return formatter.write_df_category()

    def _generate_unique_columns(self, columns):
        text_df_columns = self.text_df.columns.tolist()
        return [generate_unique(column, text_df_columns) for column in columns]

    def _format_no_category(self) -> pd.DataFrame:
        self._match_no_category(self.language)
        if self.mode == OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value:
            formatter = self.instanciate_class(FormatterByDocumentJson)
            formatter.tag_columns = ["tag_json_full"]
        if self.mode == OutputFormat.ONE_ROW_PER_DOCUMENT.value:
            formatter = self.instanciate_class(FormatterByDocument)
            formatter.tag_columns = self._generate_unique_columns(
                [
                    "tag_keywords",
                    "tag_sentences",
                    "tag_list",
                ]
            )
        if self.mode == OutputFormat.ONE_ROW_PER_TAG.value:
            formatter = self.instanciate_class(FormatterByTag)
            formatter.tag_columns = self._generate_unique_columns(
                ["tag_keyword", "tag_sentence", "tag"]
            )
        return formatter.write_df()

    def instanciate_class(self, formatter):
        return formatter(
            self.text_df,
            self.splitted_sentences_column,
            self.nlp_dict,
            self.matcher_dict,
            self.text_column,
            self.language,
            self.keyword_to_tag,
            self.category_column,
        )

    def tag_and_format(self) -> pd.DataFrame:
        """
        Public function called in tagger.py
        Uses a spacy pipeline
        -Split sentences by applying sentencizer on documents
        -Uses the right Matcher depending on the presence of categories
        -nlp_dict and matcher_dict are dictionaries with language_codes for keys
        (-> Storage is in dictionary structures in prevision of the Multilingual implementation)
        """
        tokenizer = MultilingualTokenizer(split_sentences=True)
        # multilingual case
        if self.language == "language_column":
            raise NotImplementedError(
                "The multilingual mode has not been implemented yet."
            )
        # monolingual case
        else:
            # sentence splitting
            logging.info(f"Splitting sentences on {len(self.text_df)} documents...")
            start = perf_counter()
            tokenizer._add_spacy_tokenizer(self.language)
            self.nlp_dict = tokenizer.spacy_nlp_dict
            self.splitted_sentences_column = generate_unique(
                "list_sentences", self.text_df.columns.tolist()
            )
            self.text_df[self.splitted_sentences_column] = self.text_df.apply(
                self._list_sentences, axis=1
            )
            logging.info(
                f"Splitting sentences on {len(self.text_df)} documents: Done in {perf_counter() - start:.2f} seconds"
            )
            # matching
            self._get_patterns()
            logging.info(f"Tagging {len(self.text_df)} documents...")
            if self.category_column:
                return self._format_with_category()
            else:
                return self._format_no_category()