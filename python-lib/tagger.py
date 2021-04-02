"""Main module to tag the documents"""

from spacy_tokenizer import MultilingualTokenizer
from dkulib_io_utils import generate_unique
from spacy.matcher import PhraseMatcher
from fastcore.utils import store_attr
from typing import AnyStr, List
import pandas as pd
from time import perf_counter
import logging
from formatter_instanciator import FormatterInstanciator


class Tagger:
    def __init__(
        self,
        text_df: pd.DataFrame,
        ontology_df: pd.DataFrame,
        text_column: AnyStr,
        language: AnyStr,
        language_column: AnyStr,
        tag_column: AnyStr,
        category_column: AnyStr,
        keyword_column: AnyStr,
        lemmatization: bool,
        case_insensitive: bool,
        normalization: bool,
        output_format: AnyStr,
    ):
        store_attr()
        self.matcher_dict = {}
        self.nlp_dict = {}

        # remove rows with missing values
        self.ontology_df.replace("", float("nan"), inplace=True)
        self.ontology_df.dropna(inplace=True)
        if self.ontology_df.empty:
            raise ValueError(
                "No valid tags were found. Please specify at least a keyword and a tag in the ontology dataset, and re-run the recipe"
            )

    def _get_patterns(self) -> None:
        """
        Public function called in tagger.py
        Creates the list of patterns :
        - If there aren't category -> list of the keywords (string list)
        - If there are categories  -> list of dictionaries, {"label": category, "pattern": keyword}
        """
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

    def get_formatter_config(self):
        return {
            "input_df": self.text_df,
            "splitted_sentences_column": self.splitted_sentences_column,
            "nlp_dict": self.nlp_dict,
            "matcher_dict": self.matcher_dict,
            "text_column": self.text_column,
            "language": self.language,
            "keyword_to_tag": self.keyword_to_tag,
            "category_column": self.category_column,
        }

    def _format_with_category(self, arguments) -> pd.DataFrame:
        formatter = FormatterInstanciator().get_formatter(
            arguments, self.output_format, "category"
        )
        return formatter.write_df_category()

    def _generate_unique_columns(self, columns):
        text_df_columns = self.text_df.columns.tolist()
        return [generate_unique(column, text_df_columns) for column in columns]

    def _format_no_category(self, arguments) -> pd.DataFrame:
        formatter = FormatterInstanciator().get_formatter(
            arguments, self.output_format, "no_category"
        )
        return formatter.write_df()

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
            formatter_config = self.get_formatter_config()
            if self.category_column:
                self._match_with_category(self.language)
                return self._format_with_category(formatter_config)
            else:
                self._match_no_category(self.language)
                return self._format_no_category(formatter_config)