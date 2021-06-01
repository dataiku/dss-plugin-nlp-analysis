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
    if prefix:
        name_base = f"{prefix}_{name}"
    else:
        name_base = name
    if name_base not in existing_names:
        return name_base
    for j in range(1, 1001):
        new_name = f"{name_base}_{j}"
        if new_name not in existing_names:
            return new_name
    raise RuntimeError(f"Failed to generated a unique name for '{name}'")


def generate_unique_columns(
    df: pd.DataFrame, columns: List[AnyStr], prefix=None
) -> List[AnyStr]:
    """Generate unique names for columns to add in a dataframe"""
    df_columns = df.columns.tolist()
    return [
        generate_unique(name=column, existing_names=df_columns, prefix=prefix)
        for column in columns
    ]


def move_columns_after(
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
    input_df_columns = [
        column for column in df.columns.tolist() if column not in columns_to_move
    ]
    after_column_position = input_df_columns.index(after_column) + 1
    reordered_columns = (
        input_df_columns[:after_column_position]
        + columns_to_move
        + input_df_columns[after_column_position:]
    )
    return df.reindex(columns=reordered_columns)


def replace_nan_values(df: pd.DataFrame, columns_to_clean: List) -> pd.DataFrame:
    """"Clean a pandas.DataFrame to replace NaNs values by empty strings in the columns_to_clean columns of the dataframe"""
    for column in columns_to_clean:
        df[column] = df[column].fillna("")
    return df
