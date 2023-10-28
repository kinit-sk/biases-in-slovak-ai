
import json
import os
import pandas as pd
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from translators.translator import Translator


class NLLB(Translator):
    """
    A class for creating a translator model using the NLLB200 model.
    """

    language_map = {
        'be': 'bel_Cyrl',
        'cs': 'ces_Latn',
        'hr': 'hrv_Latn',
        'pl': 'pol_Latn',
        'ru': 'rus_Cyrl',
        'sk': 'slk_Latn',
        'sl': 'slv_Latn',
        'sr': 'srp_Cyrl',
        'uk': 'ukr_Cyrl',
    }
     
    def __init__(self, dir_path, target_language, variant='3.3B', device=0): 
        self.dir_path = dir_path
        super().__init__(target_language)
        self.target_language = self.language_map[target_language]
        self.model_name = self.get_model_name(variant)
        self.device = device

    
    def load(self):
        """
        Loads the model and returns an instance of the class.
        """
        super().load()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            'translation',
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang='eng_Latn',
            tgt_lang=self.target_language,
            device=self.device,
            max_length=512,
            no_repeat_ngram_size=3,
        )
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
        result = self.pipeline(text, max_length=512)
        return result[0]['translation_text']
