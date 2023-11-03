
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

    dir_path = os.path.join('cache', 'translations', 'nllb_3b')
    
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
     
    def __init__(self, target_language, variant='3.3B', device='cpu', enable_inference=False): 
        super().__init__(target_language)
        self.target_language = self.language_map[target_language]
        self.model_name = self.get_model_name(variant)
        self.device = device
        self.enable_inference = enable_inference

        if self.enable_inference:
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
        if not self.enable_inference:
            raise RuntimeError(f'This translator object does not have inference enabled. Use `NLLB(enable_inference=True)` if you wish to run inference. You tried to translate: "{text}"')
            
        result = self.pipeline(text, max_length=512)
        return result[0]['translation_text']
