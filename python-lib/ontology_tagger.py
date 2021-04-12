"""Main module to tag the documents"""
from spacy_tokenizer import MultilingualTokenizer
from plugin_io_utils import generate_unique
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
        ontology_df: pd.DataFrame,
        tag_column: AnyStr,
        category_column: AnyStr,
        keyword_column: AnyStr,
        language: AnyStr,
        lemmatization: bool,
        case_insensitivity: bool,
        normalization: bool,
    ):
        store_attr()
        self.matcher_dict = {}
        self.nlp_dict = {}
        self.keyword_to_tag = {}
        # remove rows with missing values
        self.ontology_df.replace("", float("nan"), inplace=True)
        self.ontology_df.dropna(inplace=True)
        if self.ontology_df.empty:
            raise ValueError(
                "No valid tags were found. Please specify at least a keyword and a tag in the ontology dataset, and re-run the recipe"
            )
    
    def _generate_unique_columns(self, text_df, columns):
        """Generate unique names for the new columns to add"""
        text_df_columns = text_df.columns.tolist()
        return [generate_unique(column, text_df_columns) for column in columns]

    def _split_sentences(self, text_df, text_column, language_column):
        """Split each document into a list of sentences"""
        if self.language == "language_column":
            return text_df.apply(
                self._list_sentences_multilingual,
                args=[text_column, language_column],
                axis=1,
            )
        else:
            return text_df.apply(self._list_sentences, args=[text_column], axis=1)

    def _list_sentences(self, row: pd.Series, text_column: AnyStr) -> List[AnyStr]:
        """Called if there is only one language specified
        Applies sentencizer and return list of sentences"""
        document = row[text_column]
        if type(document) != str: return []
        return [
            sentence.text for sentence in self.nlp_dict[self.language](document).sents
        ]

    def _list_sentences_multilingual(
        self, row: pd.Series, text_column: AnyStr, language_column: AnyStr
    ) -> List[AnyStr]:
        """
        Called if there are multiple languages in the document dataset
        Applies sentencizer and return list of sentences"""
        document = row[text_column]
        if type(document) != str: return []
        return [
                sentence.text
                for sentence in self.nlp_dict[row[language_column]](document).sents
            ]


    def _get_patterns(self, list_of_keywords) -> List:
        """
        Creates the list of patterns :
        - If there aren't category -> list of the keywords (string list)
        - If there are categories  -> list of dictionaries, {"label": category, "pattern": keyword}
        """
        if self.category_column:
            list_of_categories = self.ontology_df[self.category_column].values.tolist()
            return [
                {"label": label, "pattern": pattern}
                for label, pattern in zip(list_of_categories, list_of_keywords)
            ]
        else:
            return list_of_keywords

    def _tokenize_keywords(self, language, tags, keywords):
        """
        Fill in the dictionary keyword_to_tag
        The keywords are tokenized depending on the given language
        """
        tokenized_keywords = list(self.nlp_dict[language].tokenizer.pipe(keywords))
        self.keyword_to_tag[language] = {
            keyword.text: tag for keyword, tag in zip(tokenized_keywords, tags)
        }
        return tokenized_keywords
    
    def get_formatter_config(self):
        """Returns a dictionary containing the arguments to pass to the Formatter"""
        return {
            "language": self.language,
            "splitted_sentences_column": self.splitted_sentences_column,
            "nlp_dict": self.nlp_dict,
            "matcher_dict": self.matcher_dict,
            "keyword_to_tag": self.keyword_to_tag,
            "category_column": self.category_column,
        }
    
    def _match_with_category(self, patterns, list_of_tags, list_of_keywords) -> None:
        """Tokenizes keywords for every language
        Instanciates EntityRuler with associated tags and categories"""
        for language in self.nlp_dict:
            self._tokenize_keywords(language, list_of_tags, list_of_keywords)
            self.nlp_dict[language].remove_pipe("sentencizer")
            ruler = self.nlp_dict[language].add_pipe("entity_ruler")
            ruler.add_patterns(patterns)

    def _format_with_category(
        self, arguments, text_df, text_column, output_format, language_column
    ) -> pd.DataFrame:
        """Instanciate formatter and return the created output dataframe, when there are categories"""
        formatter = FormatterInstanciator().get_formatter(
            arguments, output_format, "category"
        )
        formatter.tag_columns = self._generate_unique_columns(
            text_df, formatter.tag_columns
        )
        return formatter.write_df_category(text_df, text_column, language_column)

    def _match_no_category(self, patterns, list_of_tags, list_of_keywords) -> None:
        """
        Tokenizes keywords for every language
        Instanciates PhraseMatcher with associated tags"""
        for language in self.nlp_dict:
            patterns = self._tokenize_keywords(language, list_of_tags, list_of_keywords)
            self.nlp_dict[language].remove_pipe("sentencizer")
            matcher = PhraseMatcher(self.nlp_dict[language].vocab)
            matcher.add("PatternList", patterns)
            self.matcher_dict[language] = matcher

    def _format_no_category(
        self, arguments, text_df, text_column, output_format, language_column
    ) -> pd.DataFrame:
        """Instanciate formatter and return the created output dataframe, when there is no category"""
        formatter = FormatterInstanciator().get_formatter(
            arguments, output_format, "no_category"
        )
        formatter.tag_columns = self._generate_unique_columns(
            text_df, formatter.tag_columns
        )
        return formatter.write_df(text_df, text_column, language_column)

    def tag_and_format(
        self, text_df, text_column, language_column, output_format,languages
    ) -> pd.DataFrame:
        """
        Public function called in recipe.py
        Uses a spacy pipeline
        -Split sentences by applying sentencizer on documents
        -Uses the right Matcher depending on the presence of categories
        """
        tokenizer = MultilingualTokenizer(use_models=True,split_sentences=True)
        logging.info(f"Splitting sentences on {len(text_df)} documents...")
        start = perf_counter()
        # creating a dictionary of nlp objects, one per language
        for language in languages:
            tokenizer._add_spacy_tokenizer(language)
        self.nlp_dict = tokenizer.spacy_nlp_dict
        # splitting sentences
        self.splitted_sentences_column = generate_unique(
            "list_sentences", text_df.columns.tolist()
        )
        text_df[self.splitted_sentences_column] = self._split_sentences(
            text_df, text_column, language_column
        )
        logging.info(
            f"Splitting sentences on {len(text_df)} documents: Done in {perf_counter() - start:.2f} seconds"
        )
        # matching and formatting
        list_of_tags = self.ontology_df[self.tag_column].values.tolist()
        list_of_keywords = self.ontology_df[self.keyword_column].values.tolist()
        # patterns to add to Phrase Matcher/Entity Ruler pipe
        patterns = self._get_patterns(list_of_keywords)
        formatter_config = self.get_formatter_config()
        logging.info(f"Tagging {len(text_df)} documents...")
        if self.category_column:
            self._match_with_category(patterns, list_of_tags, list_of_keywords)
            return self._format_with_category(
                formatter_config, text_df, text_column, output_format, language_column
            )
        else:
            self._match_no_category(patterns, list_of_tags, list_of_keywords)
            return self._format_no_category(
                formatter_config, text_df, text_column, output_format, language_column
            )