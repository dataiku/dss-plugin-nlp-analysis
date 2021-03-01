# -*- coding: utf-8 -*-
from dataiku.customrecipe import *
import dataiku
import pandas as pd
from dataiku import pandasutils as pdu
from dataiku.customrecipe import get_recipe_config
from tagger import Tagger

#All plugin parameters name
MANDATORY_PARAMS = ['text_column',
                    'language_column',
                    'tag_column',
                    'case_sensitivity',
                    'lemmatization',
                    'unicode_normalization',
                    'possible_outputs']

OPTIONAL_PARAMS  = ['keyword_column',
                   'category_column']


#raise an error when called
class ParameterError(ValueError):
    pass

#get input datasets
def get_input_datasets(role):
    return dataiku.Dataset(get_input_names_for_role(role)[0])

def get_output_datasets(role):
    return dataiku.Dataset(get_output_names_for_role(role)[0])

def load_settings():
    settings = {}

    
    #get inputs 
    settings['text_input']     = get_input_datasets('documents_to_tag').get_dataframe()
    settings['ontology_input'] = get_input_datasets('ontology_of_tags').get_dataframe()

    #get parameters
    for par in MANDATORY_PARAMS:
        try:
            settings[par] = get_recipe_config()[par]
        except:
            raise ParameterError(
                'Missing input column : {} '.format(par)
            )       
    
    key_col = get_recipe_config().get('keyword_column')
    settings['keyword_column'] = key_col if key_col else None
     
    cat_column = get_recipe_config().get('category_column')
    settings['category_column'] = cat_column if len(cat_column)>0 else None 
        
    return settings

#check if the given columns come from the right input datasets
def check_columns(settings):
    
    input_text_cols   = settings['text_input'].columns
    input_onto_cols   = settings['ontology_input'].columns
    
    input_text_params = [ 'text_column',
                          'language_column']
    input_onto_params = ['tag_column',
                         'keyword_column',
                         'category_column']
    
    for col in input_text_params:
        if settings[col] not in input_text_cols:
            raise ParameterError(
                'Invalid input column : {}'.format(col)
            )
    for col in input_onto_params:
        if settings[col] is not None and settings[col] not in input_onto_cols:
            raise ParameterError(
            'Invalid input column : {}'.format(settings[col])
            )

#give settings to the Tagger and get the created output dataset   #TODO decomment when adding class Tagger in python-lib
#def call_tagger(settings):
#    tagging = Tagger(settings)
#    df      = tagging.tagging_proceedure()
#    return df

            
def process_params():
    
    settings                = load_settings()
    check_columns(settings)
    tagged_documents_df     = pd.DataFrame()
    #tagged_documents_df     = call_tagger(settings) #TODO decomment when adding class Tagger in python-lib 
    
    #write output dataset
    output_ds               = get_output_datasets('tagged_documents')
    output_ds.write_with_schema(tagged_documents_df)

#run
process_params()
