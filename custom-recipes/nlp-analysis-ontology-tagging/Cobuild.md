# Cobuild guidance

Use this recipe to enrich text documents with ontology-defined tags by matching keywords in the text.

Roles:
- `document_dataset`: required input dataset containing the documents to tag.
- `ontology_dataset`: required input dataset mapping each keyword to a tag and, optionally, a category.
- `tagged_documents`: required existing output dataset for the enriched documents.

Core configuration:
- Set `text_column` to the text column in `document_dataset`.
- Set `keyword_column` and `tag_column` to the corresponding columns in `ontology_dataset`.
- Set `category_column` when the ontology groups tags into categories; otherwise leave it empty.
- Set `language` to the document language when all documents use the same language.
- Set `language=language_column` and then set `language_column` to a column containing ISO 639-1 language codes when documents are multilingual.

Matching:
- Set `ignore_case=true` when keyword matching should be case-insensitive.
- Set `ignore_diacritics=true` when accents and other diacritics should not affect matching.
- Set `lemmatization` only when it is visible for the selected single language and the user wants inflected forms to match.
- Set `lemmatization_multilingual` only when `language=language_column` and multilingual lemmatization is requested.

Output:
- Use `one_row_per_doc` to keep one row per document with array-style tag columns.
- Use `one_row_per_doc_json` to keep one row per document with JSON-style tag data.
- Use `one_row_per_match` to emit a separate row for each matched keyword.
