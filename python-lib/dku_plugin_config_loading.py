from dataiku.customrecipe import *
import dataiku
import pandas as pd
from dataiku import pandasutils as pdu
from dataiku.customrecipe import get_recipe_config
from language_dict import SUPPORTED_LANGUAGES_SPACY
from dku_config import DkuConfig

#All plugin parameters name
MATCHING_PARAMS       = ['case_sensitivity',
                         'lemmatization',
                         'unicode_normalization'
                        ]

def get_input_datasets(role):
    return dataiku.Dataset(get_input_names_for_role(role)[0])

def load_settings():
    
    text_input        = get_input_datasets('Document dataset').get_dataframe()
    ontology_input    = get_input_datasets('Ontology dataset').get_dataframe()

    config=get_recipe_config()
    
    
    input_text_cols   = text_input.columns
    input_onto_cols   = ontology_input.columns
    
    dku_config = DkuConfig(
        local_vars    = dataiku.Project().get_variables()['local'],
        local_prefix  = "Ontology-tagging"
    )

    dku_config.add_param(
        name          = "text_input",
        value         = text_input,
        required      = True
    )

    dku_config.add_param(
        name          = "ontology_input",
        value         = ontology_input,
        required      = True
    )
    
    for par in MATCHING_PARAMS:
            dku_config.add_param(
            name     = par,
            value    = config[par],
            required = True
            )
     
    text_col         = config.get("text_column")
    dku_config.add_param(
    name             = "text_column",
    value            = text_col,
    required         = True,
    checks           = [
        {
            'type'   : 'exists',
            'err_msg': 'Missing input column : {}.\n'.format("Text_column")
        },
        {   'type'   : 'custom',
            'cond'   : text_col in input_text_cols,
            'err_msg': 'Invalid input column : {}.\n'.format("Text column")
        }])
    
    dku_config.add_param(
    name             = "language",
    value            = config.get('language'),
    required         = True,
    checks           = [
        {
            'type'   : 'exists',
            'err_msg': 'You must select one of the languages.\n'
                       'If your dataset contains several languages, you can use "Language column" and create a column in your Document dataset containing the languages of the documents.\n'
        },
        {
            'type'   : 'in',
            'op'     : list(SUPPORTED_LANGUAGES_SPACY.keys()) + ['language_column'],
            'err_msg': 'You must select one of the languages.\n'
                       'If your dataset contains several languages, you can use "Language column" and create a column in your Document dataset containing the languages of the documents.'
        }])
    
    lang_col         = config.get('language_column')
    dku_config.add_param(
    name             = "language_column",
    value            = lang_col if lang_col else None,
    required         = (dku_config.language == 'language_column'),
    checks           = [
        {
            'type'   : "custom",
            'cond'   : lang_col != None,
            'err_msg': 'Missing input column : {}.\n'.format("Language column")
        },
        
        {
            'type'   : 'custom',
            'cond'   : lang_col in input_text_cols,
            'err_msg': 'Invalid input column : {}.\n'.format("Language column")
        }])
    
    tag_col          = config.get("tag_column")
    dku_config.add_param(
    name             = "tag_column",
    value            = tag_col,
    required         = True,
    checks           = [
        {
            'type'   : 'exists',
            'err_msg': 'Missing input column : {}.\n'.format("Tag column")
        },
        {   'type'   : 'custom',
            'cond'   : tag_col in input_onto_cols,
            'err_msg': 'Invalid input column : {}.\n'.format("Tag column")
        }])
    
    key_col          = config.get("keyword_column")
    dku_config.add_param(
    name             = "keyword_column",
    value            = key_col,
    required         = False,
    checks           = [
        {
            'type'   : 'custom',
            'cond'   : key_col in input_onto_cols or key_col==None,
            'err_msg': 'Invalid input column : {}.\n'.format("Keyword column")
        }])
    
    cat_col          = config.get("category_column")
    dku_config.add_param(
    name             = "category_column",
    value            = cat_col,
    required         = False,
    checks           = [
        {
            'type'   :'custom',
            'cond'   : cat_col in input_onto_cols or cat_col==None,
            'err_msg': 'Invalid input column : {}.\n'.format("Category column")
        }])
    
    output           = config.get("output_format")
    dku_config.add_param(
    name             = "output_format",
    value            = output,
    required         = dku_config.category_column != None,
    checks           = [
        {
            'type'   : 'exists',
            'err_msg': 'Missing parameter : {}'.format('Output format')
        }])
    
    output           = config.get("output_format_with_categories")
    dku_config.add_param(
    name             = "output_format",
    value            = output,
    required         = dku_config.category_column != None,
    checks           = [
        {
            'type': 'exists',
            'err_msg': 'Missing parameter : {}'.format('Output format')
        }])
    
    return dku_config