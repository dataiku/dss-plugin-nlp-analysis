# Cobuild guidance

Runtime constraints:
- When `language=language_column`, configure `language_column`; the runner requires it even though the descriptor does not mark the parameter mandatory.
- Every non-null value in `language_column` must be a supported language code listed by the descriptor.
