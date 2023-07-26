
import json
import os
import pandas as pd
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from translators.translator import Translator


class NLLB200(Translator):
    """
    A class for creating a translator model using the NLLB200 model.

    """
     
    def __init__(self, data_path, variant='3.3B', device=0): 
        super().__init__(data_path)
        self.model_name = self.get_model_name(variant)
        self.device = device

    
    def load(self):
        """
        Loads the model and returns an instance of the class.
        """
        super().load()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.loaded = True
        return self

        
    def get_model_name(self, variant):
        """
        This function retrieves the model name of a NLLB200 model based on the specified variant.
        """
        variants = {
            '600M': 'distilled-600M',
            '1.3B': 'distilled-1.3B',
            '3.3B': '3.3B'
        }
        return f'facebook/nllb-200-{variants[variant]}'
        
        
    def _call_translation(self, text):
        """
        Translates batch of texts and returns the translated batch of texts.
        """                  

        translation_pipeline = pipeline(
            'translation',
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang='eng_Latn',
            tgt_lang='slk_Latn',
            device=self.device,
            max_length=512,
            no_repeat_ngram_size=3
        )
        

        result = translation_pipeline(text, max_length=512)
        return result[0]['translation_text']
