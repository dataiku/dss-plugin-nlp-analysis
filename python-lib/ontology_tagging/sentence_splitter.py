import pandas as pd
import logging
from time import perf_counter
from tqdm import tqdm

from typing import AnyStr, List, Tuple

from spacy.tokens import Doc
from fastcore.utils import store_attr

from utils.plugin_io_utils import replace_nan_values, generate_unique


class SentenceSplitter:
    """Module to handle sentence splitting with spaCy 'sentencizer' for multiple languages

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
        language,
        language_column=None,
    ):
        store_attr()

    @staticmethod
    def _clean_linebreaks(text: AnyStr) -> AnyStr:
        """Replace multiple end-of-line characters and whitespaces with single ones"""
        text= "\n".join(line.strip() for line in filter(None, text.splitlines()))
        # filter() is used to remove the whitespace characters in case there are multiple line breaks ("\n\n")
        return text

    def split_sentences_df(self) -> Tuple[pd.DataFrame, AnyStr]:
        """Append new column(s) to a dataframe, with documents as lists of sentences

        Returns:
            pandas.DataFrame : text_df with the new added column(s) of tokenized text
            str : Name of the new column tokenized column

        """
        # clean NaN documents before splitting
        self.text_df = replace_nan_values(
            df=self.text_df, columns_to_clean=[self.text_column]
        )
        # generate a unique name for the column of tokenized text
        start = perf_counter()
        # split sentences with spacy sentencizer
        text_column_tokenized = generate_unique(
            name="list_sentences", existing_names=self.text_df.columns.tolist()
        )
        logging.info(f"Splitting sentences on {len(self.text_df)} documents...")
        self.text_df[text_column_tokenized] = self._get_splitted_sentences()
        logging.info(
            f"Splitting sentences on {len(self.text_df)} documents: Done in {perf_counter() - start:.2f} seconds"
        )
        return self.text_df, text_column_tokenized

    def _split_sentences_multilingual(self, row: pd.Series) -> List[AnyStr]:
        """Called if there are multiple languages in the document dataset. Apply sentencizer and return list of sentences

        Args:
            row (pandas.DataFrame): row which contains the text to split

        Returns:
            List: Document splitted into sentences as strings.

        """
        document = self._clean_linebreaks(row[self.text_column])
        language = row[self.language_column]
        return [
            sentence.text
            for sentence in self.tokenizer.spacy_nlp_dict[language](document).sents
        ]

    def _split_sentences(self, row: pd.Series) -> List[AnyStr]:
        """Called if there is only one language specified.Apply sentencizer and return list of sentences

        Args:
            row (pandas.Series): row which contains text to process

        Returns:
            List : Document splitted into tokenized sentences as strings.

        """
        document = self._clean_linebreaks(row[self.text_column])
        return [
            sentence.text
            for sentence in self.tokenizer.spacy_nlp_dict[self.language](document).sents
        ]

    def _get_splitted_sentences(self) -> pd.DataFrame:
        """Call either _split_sentences or _split_sentences_multilingual

        Returns:
            pandas.DataFrame: dataframe with the new tokenized text column

        """
        tqdm.pandas(miniters=1, mininterval=5.0)
        if self.language_column:
            return self.text_df.progress_apply(
                self._split_sentences_multilingual,
                axis=1,
            )
        else:
            return self.text_df.progress_apply(self._split_sentences, axis=1)
