import dataiku
from typing import Dict


def set_column_descriptions(
    output_dataset: dataiku.Dataset,
    column_descriptions: Dict,
    input_dataset: dataiku.Dataset = None,
) -> None:
    """Set column descriptions of the output dataset based on a dictionary of column descriptions
    Retain the column descriptions from the input dataset if the column name matches.
    Args:
        output_dataset: Output dataiku.Dataset instance
        column_descriptions: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions
    """
    output_dataset_schema = output_dataset.read_schema()
    input_dataset_schema = []
    input_columns_names = []
    if input_dataset is not None:
        input_dataset_schema = input_dataset.read_schema()
        input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_descriptions.get(output_col_name)
        matched_comment = next(
            (
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ),
            None,
        )
        if matched_comment is not None:
            output_col_info["comment"] = matched_comment
    output_dataset.write_schema(output_dataset_schema)
