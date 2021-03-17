"""Module to prepare the tagging of the documents
splits on sentences and instanciates a Matcher to tag the documents"""

from spacy_tokenizer import MultilingualTokenizer
from dkulib_io_utils import generate_unique
from spacy.matcher import PhraseMatcher
from fastcore.utils import store_attr
from typing import AnyStr
import pandas as pd


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
        case_sensitive,
        normalize,
        mode,
    ):
        store_attr(but=["text_df", "ontology_df"])
        self.text_df = text_df.get_dataframe()
        self.ontology_df = ontology_df.get_dataframe()
        self.matcher_dict = {}
        self.nlp_dict = {}
        self._matching_pipeline()

    def _get_patterns(self):
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

    def _list_sentences(self, row: pd.Series):
        """Auxiliary function called in _matching_pipeline
        Applies sentencizer and return list of sentences"""
        return [
            sentence.text
            for sentence in self.nlp_dict[self.language](row[self.text_column]).sents
        ]

    def _matching_method_no_category(self, language: AnyStr):
        """instanciates PhraseMatcher with associated tags"""
        _, _ = self.nlp_dict[language].remove_pipe("sentencizer")
        matcher = PhraseMatcher(self.nlp_dict[language].vocab)
        self.patterns = list(self.nlp_dict[language].tokenizer.pipe(self.patterns))
        matcher.add("PatternList", self.patterns)
        self.matcher_dict[language] = matcher

    def _matching_method_with_category(self, language: AnyStr):
        """Instanciates EntityRuler with associated tags and categories"""
        self.nlp_dict[language].remove_pipe("sentencizer")
        ruler = self.nlp_dict[language].add_pipe("entity_ruler")
        ruler.add_patterns(self.patterns)

    def _matching_pipeline(self):
        """
        Public function called in tagger.py
        Uses a spacy pipeline
        -Split sentences by applying sentencizer on documents
        -Uses the right Matcher depending on the presence of categories
        -nlp_dict and matcher_dict are dictionaries with language_codes for keys
        (-> Storage is in dictionary structures in prevision of the Multilingual implementation)
        """
        tokenizer = MultilingualTokenizer()
        # multilingual case
        if self.language == "language_column":
            raise NotImplementedError(
                "The multilingual mode has not been implemented yet."
            )
        # monolingual case
        else:
            # sentence splitting
            tokenizer._add_spacy_tokenizer(self.language, True)
            self.nlp_dict = tokenizer.spacy_nlp_dict
            self.splitted_sentences_column = generate_unique(
                "list_sentences", self.text_df.columns.tolist()
            )
            self.text_df[self.splitted_sentences_column] = self.text_df.apply(
                self._list_sentences, axis=1
            )
            # matching
            self._get_patterns()
            if self.category_column:
                # precision : giving a class attribute argument because those functions will be used with non-attribute class
                # function in multilingual case
                self._matching_method_with_category(self.language)
            else:
                self._matching_method_no_category(self.language)
