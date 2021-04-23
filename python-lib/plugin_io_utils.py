# -*- coding: utf-8 -*-
"""Module with read/write utility functions which are *not* based on the Dataiku API"""
import re
import logging
import functools
from typing import List, AnyStr, Union, Callable
from time import perf_counter
import pandas as pd
import numpy as np
from spacy.tokens import Span, Doc


def unique_list(sequence: List) -> List:
    """Make a list unique, ordering values by order of appearance in the original list
    Args:
        sequence: Original list
    Returns:
       List with unique elements ordered by appearance in the original list
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def truncate_text_list(
    text_list: List[AnyStr], num_characters: int = 140
) -> List[AnyStr]:
    """Truncate a list of strings to a given number of characters
    Args:
        text_list: List of strings
        num_characters: Number of characters to truncate each string to
    Returns:
       List with truncated strings
    """
    output_text_list = []
    for text in text_list:
        if len(text) > num_characters:
            output_text_list.append(text[:num_characters] + " (...)")
        else:
            output_text_list.append(text)
    return output_text_list


def clean_text_df(
    df: pd.DataFrame, dropna_columns: List[AnyStr] = None
) -> pd.DataFrame:
    """Clean a pandas.DataFrame with text columns to remove empty strings and NaNs values in the dataframe
    Args:
        df: Input pandas.DataFrame which should contain only text
        dropna_columns: Optional list of column names where empty strings and NaN should be checked
            Default is None, which means that all columns will be checked
    Returns:
       pandas.DataFrame with rows dropped in case of empty strings or NaN values
    """
    for col in df.columns:
        df[col] = df[col].str.strip().replace("", np.NaN)
    df = df.dropna(subset=dropna_columns)
    return df


def generate_unique(
    name: AnyStr, existing_names: List[AnyStr], prefix: AnyStr = None
) -> AnyStr:
    """Generate a unique name among existing ones by suffixing a number and adding a prefix
    Args:
        name: Input name
        existing_names: List of existing names
        prefix: Optional prefix to add to the output name
    Returns:
       Unique name with a number suffix in case of conflict, and an optional prefix
    """
    name = re.sub(r"[^\x00-\x7F]", "_", name).replace(
        " ", "_"
    )  # replace non ASCII and whitespace characters by an underscore _
    if prefix:
        new_name = f"{prefix}_{name}"
    else:
        new_name = name
    for j in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = f"{new_name}_{j}"
    raise RuntimeError(f"Failed to generated a unique name for '{name}'")


def move_columns_after(
    input_df: pd.DataFrame,
    df: pd.DataFrame,
    columns_to_move: List[AnyStr],
    after_column: AnyStr,
) -> pd.DataFrame:
    """Reorder columns by moving a list of columns after another column
    Args:
        df: Input pandas.DataFrame
        columns_to_move: List of column names to move
        after_column: Name of the columns to move columns after
    Returns:
       pandas.DataFrame with reordered columns
    """
    input_df_columns = input_df.columns.tolist()
    after_column_position = input_df.columns.get_loc(after_column) + 1
    reordered_columns = (
        input_df_columns[:after_column_position]
        + columns_to_move
        + input_df_columns[after_column_position:]
    )
    return df.reindex(columns=reordered_columns)


def get_attr(case_insensitive: bool) -> AnyStr:
    """Return spaCy case-sensitivity attribute"""
    return "LOWER" if case_insensitive else "ORTH"


def get_keyword(text: AnyStr, case_insensitive: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if case_insensitive else text


def get_sentence(span: Span, case_insensitive: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if case_insensitive else span.as_doc()


def replace_nan_values(df: pd.DataFrame, columns_to_clean: List) -> pd.DataFrame:
    """"Clean a pandas.DataFrame to replace NaNs values by empty strings in the columns_to_clean columns of the dataframe"""
    for column in columns_to_clean:
        df[column] = df[column].fillna("")
    return df
