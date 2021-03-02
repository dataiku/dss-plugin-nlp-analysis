# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
from dataiku.customrecipe import *
from dataiku import pandasutils as pdu
from dku_config import DkuConfig
from dku_plugin_config_loading import load_settings



#give settings to the Tagger and get the created output dataset   
def call_tagger(settings):
    tagging             = Tagger(settings)
    df                  = tagging.tagging_proceedure()
    return df

def process_params():
    settings            = load_settings()
    tagged_documents_df = pd.DataFrame()
    #tagged_documents_df = call_tagger(settings)    TODO decomment when using Tagger
    
    
    #write output
    output_ds = get_output_datasets('Tagged documents')
    output_ds.write_with_schema(tagged_documents_df)
    
#run
process_params()
