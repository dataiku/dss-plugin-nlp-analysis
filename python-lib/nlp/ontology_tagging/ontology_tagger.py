import pandas as pd
import logging

from fastcore.utils import store_attr
from typing import AnyStr
from typing import List

from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher
from spacy.pipeline.sentencizer import Sentencizer

from nlp.spacy_tokenizer import MultilingualTokenizer
from .formatting.instanciator import FormatterInstanciator
from nlp.sentence_splitter import SentenceSplitter

from nlp.utils import (
    lemmatize_doc,
    get_phrase_matcher_attr,
    lowercase_if,
    unicode_normalize_text,
)
from nlp.language_support import (
    SPACY_LANGUAGE_LOOKUP,
    SPACY_LANGUAGE_RULES,
    SPACY_LANGUAGE_MODELS_LEMMATIZATION,
)


class Tagger:
    """Tag text data with a given ontology. Relies on spaCy Language components:
    - Sentencizer to split document into sentences
    - PhraseMatcher and EntityRuler to tag documents

    Attributes:
        ontology_df (pandas dataframe): Ontology which contains the tags to assign
            Example:
            tag       | keyword   | category
            President | Trump     | Politics
            President | Bush      | Politics
            Democrats | Party     | Politics
            State     | California| United States

        tag_column (string): Name of the column in the Ontology. Contains the tags to assign
        category_column (string): Name of the column in the Ontology. Contains the category of each tag to assign.
        keyword_column (string): Name of column in the Ontology. Contains the keywords to match with in documents.
        language (string): language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages
            Used if there is only one language to treat.
            Use the argument 'language_column' for passing a language column name in 'tag_and_format' method otherwise.
        lemmatization (bool): If True , match on lemmatized forms. Default is False.
        ignore_case (bool): If True, match on lowercased forms. Default is False.
        ignore_diacritics (bool): If True, ignore diacritic marks e.g., accents, cedillas, tildes for matching. Default is False.
        tokenizer (MultilingualTokenizer): Tokenizer instance to create the tokenizers for each language

    """

    def __init__(
        self,
        ontology_df: pd.DataFrame,
        tag_column: AnyStr,
        category_column: AnyStr,
        keyword_column: AnyStr,
        language: AnyStr,
        lemmatization: bool = False,
        ignore_case: bool = False,
        ignore_diacritics: bool = False,
    ):
        store_attr()
        self._remove_incomplete_rows()
        if self.category_column:
            self._replace_missing_categories()
        # set the punctuation characters to use for sentence splitting
        config = {
            "sentencizer": {"punct_chars": Sentencizer.default_punct_chars + ["\n"]}
        }
        self.tokenizer = MultilingualTokenizer(
            add_pipe_components=["sentencizer"],
            enable_pipe_components="sentencizer",
            config=config,
        )
        self._matcher_dict = {}
        # Dictionary of spaCy PhraseMatcher objects filled by the _match_no_category method.
        # Unused if we are using EntityRuler (in case there are categories in the Ontology)
        self.column_descriptions = {}
        # Dictionary of new columns to add in the dataframe (key) and their descriptions (value).
        # It is filled by the _format_with_category / _format_no_category methods
        self._use_nfc = self.lemmatization and not self.ignore_diacritics
        # Text will be normalized with NFC if True, with NFD otherwise.
        self._keyword_to_tag = {}
        # Dictionary of keywords (key) and tags (value) to retrieve the tags from the matched keywords, filled by the _tokenize_keywords method.
        #    Unused if we are using EntityRuler (in case there are categories in the Ontology)
        #    Example :
        #        {"Donald Trump": "Politics", "N.Y.C" : "United States, "NBC": "News"}

    def tag_and_format(
        self,
        text_df: pd.DataFrame,
        text_column: AnyStr,
        output_format: AnyStr,
        languages: List[AnyStr],
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Public function called in recipe.py. Uses a spacy Language object to:
            -Split sentences by applying sentencizer on documents (by calling the SentenceSplitter module)
            -Use the right Matcher depending on the presence of categories (PhraseMatcher / EntityRuler) to match documents with tags
            -Write the found matches into a new DataFrame (by calling the Formatter module)

        """
        self._initialize_tokenizer(languages)
        text_df, text_column_tokenized = self._sentence_splitting(
            text_df, text_column, language_column
        )
        list_of_tags = self.ontology_df[self.tag_column].values.tolist()
        list_of_keywords = self.ontology_df[self.keyword_column].values.tolist()
        formatter_config = self.get_formatter_config(text_column_tokenized)
        # matching and formatting
        if self.category_column:
            # patterns to add to Entity Ruler pipe
            self._match_with_category(list_of_tags, list_of_keywords)
            logging.info(f"Tagging {len(text_df)} documents...")
            return self._format_with_category(
                arguments=formatter_config,
                text_df=text_df,
                text_column=text_column,
                output_format=output_format,
                language_column=language_column,
            )
        else:
            self._match_no_category(list_of_tags, list_of_keywords)
            logging.info(f"Tagging {len(text_df)} documents...")
            return self._format_no_category(
                arguments=formatter_config,
                text_df=text_df,
                text_column=text_column,
                output_format=output_format,
                language_column=language_column,
            )

    def _set_log_level(self, languages: List[AnyStr]) -> None:
        """Set Spacy log level to ERROR to hide unwanted warnings"""
        if any([item in languages for item in SPACY_LANGUAGE_RULES]):
            logger = logging.getLogger("spacy")
            logger.setLevel(logging.ERROR)

    def _remove_incomplete_rows(self) -> None:
        """Remove rows with at least one empty value in the keyword column and the tag column from ontology df"""
        self.ontology_df.replace("", float("nan"), inplace=True)
        self.ontology_df.dropna(
            subset=[self.keyword_column, self.tag_column], inplace=True
        )
        if self.ontology_df[[self.keyword_column, self.tag_column]].empty:
            raise ValueError(
                "No valid tags were found. Please specify at least a keyword and a tag in the ontology dataset, and re-run the recipe"
            )

    def _replace_missing_categories(self) -> None:
        """Replace each empty category by a category 'uncategorized'"""
        self.ontology_df[self.category_column] = self.ontology_df[
            self.category_column
        ].fillna("uncategorized")

    def _get_patterns(
        self,
        list_of_keywords: List[AnyStr],
        list_of_tags: List[AnyStr],
        language: AnyStr,
    ) -> List[dict]:
        """Called in _tag_and_format, when self.category_column is not None. Create the list of patterns to match with.

        Args:
            list_of_keywords (List): The keywords in the Ontology Dataset
            list_of_tags (List): The tags in the Ontology Dataset

        Returns:
            List : List of dictionaries. One dictionary = one pattern, defined as follow : {"label": category, "pattern": keyword, "id": tag}

        """
        if self.lemmatization:
            self.tokenizer._activate_components_to_lemmatize(language)
        list_of_categories = self.ontology_df[self.category_column].values.tolist()
        return [
            {
                "label": label,
                "pattern": unicode_normalize_text(
                    text=lowercase_if(text=pattern, lowercase=self.ignore_case),
                    use_nfc=self._use_nfc,
                    ignore_diacritics=self.ignore_diacritics,
                ),
                "id": tag,
            }
            for label, pattern, tag in zip(
                list_of_categories, list_of_keywords, list_of_tags
            )
        ]

    def _tokenize_keywords(
        self, language: AnyStr, tags: List[AnyStr], keywords: List[AnyStr]
    ) -> List[Doc]:
        """Called when self.category_column is not None. Tokenize the keywords and fill in the dictionary _keyword_to_tag.
        The keywords are tokenized depending on the given language.

        Args:
            language (str) : Language code in ISO 639-1 format to use to tokenize the keywords.
            tags (List): The tags in the Ontology Dataset.
            keywords (List): The keywords in the Ontology Dataset to tokenize.

        Returns:
            List: The tokenized keywords.

        """
        if self.lemmatization:
            self.tokenizer._activate_components_to_lemmatize(language)
        keywords = [
            unicode_normalize_text(
                text=lowercase_if(text=keyword, lowercase=self.ignore_case),
                use_nfc=self._use_nfc,
                ignore_diacritics=self.ignore_diacritics,
            )
            for keyword in keywords
        ]
        tokenized_keywords = list(
            self.tokenizer.spacy_nlp_dict[language].pipe(keywords)
        )  # list of SpaCy.tokens.Doc objects
        if self.lemmatization:
            self._keyword_to_tag[language] = {
                lemmatize_doc(doc): tag for doc, tag in zip(tokenized_keywords, tags)
            }
        else:
            self._keyword_to_tag[language] = {
                doc.text: tag for doc, tag in zip(tokenized_keywords, tags)
            }
        return tokenized_keywords

    def get_formatter_config(self, text_column_tokenized: AnyStr) -> dict:
        """Return a dictionary containing the arguments to pass to the Formatter"""
        arguments = {
            "language": self.language,
            "text_column_tokenized": text_column_tokenized,
            "tokenizer": self.tokenizer,
            "category_column": self.category_column,
            "ignore_case": self.ignore_case,
            "lemmatization": self.lemmatization,
            "ignore_diacritics": self.ignore_diacritics,
            "_use_nfc": self._use_nfc,
        }
        if not self.category_column:
            arguments["_matcher_dict"] = self._matcher_dict
            arguments["_keyword_to_tag"] = self._keyword_to_tag
        return arguments

    def _match_with_category(
        self,
        list_of_tags: List[AnyStr],
        list_of_keywords: List[AnyStr],
    ) -> None:
        """Tokenize keywords for every language. Instanciate EntityRuler with associated tags and categories"""
        for language in self.tokenizer.spacy_nlp_dict:
            patterns = self._get_patterns(list_of_keywords, list_of_tags, language)
            self.tokenizer.spacy_nlp_dict[language].remove_pipe("sentencizer")
            ruler = self.tokenizer.spacy_nlp_dict[language].add_pipe(
                "entity_ruler",
                config={
                    "phrase_matcher_attr": get_phrase_matcher_attr(self.lemmatization)
                },
            )
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
        formatter = FormatterInstanciator.get_formatter(
            config=arguments, format=output_format, category="category"
        )
        output_df = formatter.write_df_category(
            input_df=text_df,
            text_column=text_column,
            language_column=language_column,
        )
        self.column_descriptions = formatter.column_descriptions
        return output_df

    def _match_no_category(
        self,
        list_of_tags: List[AnyStr],
        list_of_keywords: List[AnyStr],
    ) -> None:
        """Tokenize keywords for every language. Instanciate PhraseMatcher with associated tags"""
        for language in self.tokenizer.spacy_nlp_dict:
            patterns = self._tokenize_keywords(language, list_of_tags, list_of_keywords)
            self.tokenizer.spacy_nlp_dict[language].remove_pipe("sentencizer")
            matcher = PhraseMatcher(
                self.tokenizer.spacy_nlp_dict[language].vocab,
                attr=get_phrase_matcher_attr(self.lemmatization),
            )
            matcher.add("PatternList", patterns)
            self._matcher_dict[language] = matcher

    def _format_no_category(
        self,
        arguments: dict,
        text_df: pd.DataFrame,
        text_column: AnyStr,
        output_format: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Instanciate formatter and return the created output dataframe, when there is no category"""
        formatter = FormatterInstanciator.get_formatter(
            config=arguments, format=output_format, category="no_category"
        )
        output_df = formatter.write_df(
            input_df=text_df,
            text_column=text_column,
            language_column=language_column,
        )
        self.column_descriptions = formatter.column_descriptions
        return output_df

    def _initialize_tokenizer(self, languages: List[AnyStr]) -> None:
        """Create a dictionary of nlp objects, one per language. Dictionary is accessible via self.tokenizer.nlp_dict"""
        if self.lemmatization:
            self._set_log_level(languages)
            self.tokenizer._set_use_models(languages)
        for language in languages:
            self.tokenizer.add_spacy_tokenizer(language)

    def _sentence_splitting(
        self, text_df, text_column, language_column=None
    ) -> pd.DataFrame:
        """Instanciate a SentenceSplitter to tokenize and split each document into sentences

        Returns:
            pandas.DataFrame : the input Dataframe text_df with additional column(s) which contains splitted sentences"""
        sentence_splitter = SentenceSplitter(
            text_df=text_df,
            text_column=text_column,
            tokenizer=self.tokenizer,
            language=self.language,
            language_column=language_column,
        )
        return sentence_splitter.split_sentences_df()
