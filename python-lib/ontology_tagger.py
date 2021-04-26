from spacy_tokenizer import MultilingualTokenizer
from formatter_instanciator import FormatterInstanciator
from plugin_io_utils import generate_unique, get_keyword
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from fastcore.utils import store_attr
from typing import AnyStr, List, Union
import pandas as pd
from time import perf_counter
import logging
from sentence_splitter import SentenceSplitter


class Tagger:
    """Tag text data with a given ontology
    Relies on spaCy Language components:
    - Sentencizer to split document into sentences
    - PhraseMatcher and EntityRuler to tag documents
    Attributes:
        ontology_df (pandas dataframe): Ontology which contains the tags to assign
        tag_column (string): Name of the column in the Ontology. Contains the tags to assign
        category_column (string): Name of the column in the Ontology. Contains the category of each tag to assign.
        keyword_column (string): Name of column in the Ontology. Contains the keywords to match with in documents.
        language (string): Name of the documents language. Used if there is only one language to treat
        lemmatization (bool): If True, match on lemmatized forms
        case_insensitivity(bool): If True, match on lowercased forms
        normalization (bool): If True, normalize diacritic marks e.g., accents, cedillas, tildes
        tokenizer (MultilingualTokenizer): Tokenizer instance to create the tokenizers for each language
        matcher_dict (dict): Dictionary of spaCy PhraseMatchers objects. Unused if we are using EntityRuler (in case there are categories in the Ontology)
        keyword_to_tag(dict): Keywords (key) and tags (value) to retrieve the tags from the matched keywords Unused if we are using EntityRuler (in case there are categories in the Ontology)
    """

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
        self._remove_incomplete_rows()
        self.tokenizer = MultilingualTokenizer(
            use_models=True,
            add_pipe_components=["sentencizer"],
            enable_pipe_components="sentencizer",
        )
        self.matcher_dict = {}  # this will be fill in the _match_no_category method
        self.keyword_to_tag = {}  # this will be fill in the _tokenize_keywords method

    def _remove_incomplete_rows(self) -> None:
        """Remove rows with at least one empty value from ontology df"""
        self.ontology_df.replace("", float("nan"), inplace=True)
        self.ontology_df.dropna(inplace=True)
        if self.ontology_df.empty:
            raise ValueError(
                "No valid tags were found. Please specify at least a keyword and a tag in the ontology dataset, and re-run the recipe"
            )

    def _generate_unique_columns(
        self, text_df: pd.DataFrame, columns: List[AnyStr]
    ) -> List[AnyStr]:
        """Generate unique names for the new columns to add"""
        text_df_columns = text_df.columns.tolist()
        return [
            generate_unique(name=column, existing_names=text_df_columns)
            for column in columns
        ]

    def _get_patterns(
        self, list_of_keywords: List[AnyStr], list_of_tags: List[AnyStr]
    ) -> Union[List[dict], List[AnyStr]]:
        """
        Create the list of patterns :
        - If there aren't category -> list of the keywords (string list)
        - If there are categories  -> list of dictionaries, {"label": category, "pattern": keyword}
        """
        list_of_categories = self.ontology_df[self.category_column].values.tolist()
        return [
            {
                "label": label,
                "pattern": get_keyword(pattern, self.case_insensitivity),
                "id": tag,
            }
            for label, pattern, tag in zip(
                list_of_categories, list_of_keywords, list_of_tags
            )
        ]

    def _tokenize_keywords(
        self, language: AnyStr, tags: List[AnyStr], keywords: List[AnyStr]
    ) -> List[Doc]:
        """
        Fill in the dictionary keyword_to_tag
        The keywords are tokenized depending on the given language
        """
        keywords = [
            get_keyword(keyword, self.case_insensitivity) for keyword in keywords
        ]
        tokenized_keywords = list(
            self.tokenizer.spacy_nlp_dict[language].tokenizer.pipe(keywords)
        )
        self.keyword_to_tag[language] = {
            get_keyword(keyword.text, self.case_insensitivity): tag
            for keyword, tag in zip(tokenized_keywords, tags)
        }
        return tokenized_keywords

    def get_formatter_config(self, tokenized_columns: List[AnyStr]) -> dict:
        """Return a dictionary containing the arguments to pass to the Formatter"""
        arguments = {
            "language": self.language,
            "text_column_tokenized": tokenized_columns[0],
            "tokenizer": self.tokenizer,
            "category_column": self.category_column,
            "case_insensitivity": self.case_insensitivity,
        }
        if self.case_insensitivity:
            arguments["text_lower_column_tokenized"] = tokenized_columns[1]
        if not self.category_column:
            arguments["matcher_dict"] = self.matcher_dict
            arguments["keyword_to_tag"] = self.keyword_to_tag
        return arguments

    def _match_with_category(
        self,
        patterns: List[dict],
        list_of_tags: List[AnyStr],
        list_of_keywords: List[AnyStr],
    ) -> None:
        """
        Tokenize keywords for every language
        Instanciate EntityRuler with associated tags and categories
        """
        for language in self.tokenizer.spacy_nlp_dict:
            self.tokenizer.spacy_nlp_dict[language].remove_pipe("sentencizer")
            ruler = self.tokenizer.spacy_nlp_dict[language].add_pipe("entity_ruler")
            ruler.add_patterns(patterns)

    def _format_with_category(
        self,
        arguments: dict,
        text_df: pd.DataFrame,
        text_column: AnyStr,
        output_format: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Instanciate formatter and return the created output dataframe, when there are categories"""
        formatter = FormatterInstanciator().get_formatter(
            config=arguments, format=output_format, category="category"
        )
        formatter.tag_columns = self._generate_unique_columns(
            text_df=text_df, columns=formatter.tag_columns
        )
        return formatter.write_df_category(
            input_df=text_df, text_column=text_column, language_column=language_column
        )

    def _match_no_category(
        self,
        list_of_tags: List[AnyStr],
        list_of_keywords: List[AnyStr],
    ) -> None:
        """
        Tokenize keywords for every language
        Instanciate PhraseMatcher with associated tags
        """
        for language in self.tokenizer.spacy_nlp_dict:
            patterns = self._tokenize_keywords(language, list_of_tags, list_of_keywords)
            self.tokenizer.spacy_nlp_dict[language].remove_pipe("sentencizer")
            matcher = PhraseMatcher(self.tokenizer.spacy_nlp_dict[language].vocab)
            matcher.add("PatternList", patterns)
            self.matcher_dict[language] = matcher

    def _format_no_category(
        self,
        arguments: dict,
        text_df: pd.DataFrame,
        text_column: AnyStr,
        output_format: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Instanciate formatter and return the created output dataframe, when there is no category"""
        formatter = FormatterInstanciator().get_formatter(
            config=arguments, format=output_format, category="no_category"
        )
        formatter.tag_columns = self._generate_unique_columns(
            text_df=text_df, columns=formatter.tag_columns
        )
        return formatter.write_df(
            input_df=text_df,
            text_column=text_column,
            language_column=language_column,
        )

    def _initialize_tokenizer(self, languages: List[AnyStr]) -> None:
        """Create a dictionary of nlp objects, one per language. Dictionary is accessible via self.tokenizer.nlp_dict"""
        for language in languages:
            self.tokenizer._add_spacy_tokenizer(language)

    def _sentence_splitting(self, text_df, text_column, language_column=None):
        sentence_splitter = SentenceSplitter(
            text_df=text_df,
            text_column=text_column,
            tokenizer=self.tokenizer,
            case_insensitivity=self.case_insensitivity,
            language=self.language,
            language_column=language_column,
        )
        return sentence_splitter._split_sentences_df()

    def tag_and_format(
        self,
        text_df: pd.DataFrame,
        text_column: AnyStr,
        output_format: AnyStr,
        languages: List[AnyStr],
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Public function called in recipe.py
        Uses a spacy pipeline
        -Split sentences by applying sentencizer on documents
        -Use the right Matcher depending on the presence of categories
        """
        self._initialize_tokenizer(languages)
        text_df, tokenized_columns = self._sentence_splitting(
            text_df, text_column, language_column
        )
        list_of_tags = self.ontology_df[self.tag_column].values.tolist()
        list_of_keywords = self.ontology_df[self.keyword_column].values.tolist()
        formatter_config = self.get_formatter_config(tokenized_columns)
        logging.info(f"Tagging {len(text_df)} documents...")
        # matching and formatting
        if self.category_column:
            # patterns to add to Entity Ruler pipe
            patterns = self._get_patterns(list_of_keywords, list_of_tags)
            self._match_with_category(patterns, list_of_tags, list_of_keywords)
            return self._format_with_category(
                arguments=formatter_config,
                text_df=text_df,
                text_column=text_column,
                output_format=output_format,
                language_column=language_column,
            )
        else:
            self._match_no_category(list_of_tags, list_of_keywords)
            return self._format_no_category(
                arguments=formatter_config,
                text_df=text_df,
                text_column=text_column,
                output_format=output_format,
                language_column=language_column,
            )