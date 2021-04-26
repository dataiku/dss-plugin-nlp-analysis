import pandas as pd
import logging
from time import perf_counter
from typing import AnyStr, List, Tuple
from spacy.tokens import Doc
from fastcore.utils import store_attr
from plugin_io_utils import replace_nan_values, generate_unique


class SentenceSplitter:
    """Module to handle sentence splitting with spaCy for multiple languages

    Attributes:
        tokenizer (dict): MultilingualTokenizer instance which stored a dictionary spacy_nlp_dict of spaCy
            Language instances (value) by language code (key)
        text_column (str): Name of the dataframe column storing documents to process
        text_df (pandas DataFrame): DataFrame which contains text_column
        language (str) : language of the documents to process.
        language_column (str) : Name of the dataframe column storing languages of the document to process.
            Default to None
        case_sensitivity (bool): Boolean used to know if the text should be resplitted with lowercase text.

    """

    def __init__(
        self,
        text_df,
        text_column,
        tokenizer,
        normalize_case,
        language,
        language_column=None,
    ):
        store_attr()

    def _split_sentences_df(self) -> Tuple[pd.DataFrame, List[AnyStr]]:
        """Append new column(s) to a dataframe, with documents as lists of sentences

        Returns:
            pandas.DataFrame : text_df with the new added column(s) of tokenized text
            List : Names of the new column(s)

        """
        # clean NaN documents before splitting
        self.text_df = replace_nan_values(
            df=self.text_df, columns_to_clean=[self.text_column]
        )
        # generate a unique name for the column of tokenized text
        logging.info(f"Splitting sentences on {len(self.text_df)} documents...")
        start = perf_counter()
        # split sentences with spacy sentencizer
        text_column_tokenized = generate_unique(
            name="list_sentences", existing_names=self.text_df.columns.tolist()
        )
        tokenized_columns = [text_column_tokenized]
        self.text_df[text_column_tokenized] = self._get_splitted_sentences()
        if self.normalize_case:
            # generate a unique name for the column of tokenized lower text
            text_lower_column_tokenized = generate_unique(
                name="text_lower", existing_names=self.text_df.columns.tolist()
            )
            tokenized_columns.append(text_lower_column_tokenized)
            # tokenize sentences in lowercase with spacy tokenizer
            self.text_df[text_lower_column_tokenized] = self.text_df.apply(
                self._lowercase_sentences,
                args=[text_column_tokenized],
                axis=1,
            )
        logging.info(
            f"Splitting sentences on {len(self.text_df)} documents: Done in {perf_counter() - start:.2f} seconds"
        )
        return self.text_df, tokenized_columns

    def _lowercase_sentences(
        self, row: pd.Series, text_column_tokenized: AnyStr
    ) -> List:
        """Retokenize text sentences by sentences in lowercase

        Args:
            row (pandas.Series): row which contains the text to process.
            text_column_tokenized (str): Name of the text column to lowercase.

        Returns :
            List : Retokenized sentences.

        """
        language = row[self.language_column] if self.language_column else self.language
        return list(
            self.tokenizer.spacy_nlp_dict[language].pipe(
                [sentence.text.lower() for sentence in row[text_column_tokenized]]
            )
        )

    def _split_sentences_multilingual(self, row: pd.Series) -> List:
        """Called if there are multiple languages in the document dataset.Apply sentencizer and return list of sentences

        Args:
            row (pandas.DataFrame): row which contains the text to split

        Returns:
            List: Document splitted into sentences.

        """
        document, language = row[self.text_column], row[self.language_column]
        splitted_sentences = self.tokenizer.spacy_nlp_dict[language](document).sents
        return list(splitted_sentences)

    def _split_sentences(self, row: pd.Series) -> List:
        """Called if there is only one language specified.Apply sentencizer and return list of sentences

        Args:
            row (pandas.Series): row which contains text to process

        Returns:
            List : Document splitted into tokenized sentences.

        """
        document = row[self.text_column]
        splitted_sentences = self.tokenizer.spacy_nlp_dict[self.language](
            document
        ).sents
        return list(splitted_sentences)

    def _get_splitted_sentences(self) -> pd.DataFrame:
        """Call either _split_sentences or _split_sentences_multilingual

        Returns:
            pandas.DataFrame: dataframe with the new tokenized column(s)

        """
        if self.language_column:
            return self.text_df.apply(
                self._split_sentences_multilingual,
                axis=1,
            )
        else:
            return self.text_df.apply(self._split_sentences, axis=1)
