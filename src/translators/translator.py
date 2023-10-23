"""
This module defines the `Translator` class, which is an abstract base class for translation models.
"""
from typing import List, Dict

import logging
import os

import pandas as pd
import tqdm
import traceback


logging.basicConfig(level=logging.INFO)


class Translator:
    """
    An abstract class for translation models.

    Currenty supported:
        `GoogleTranslate`
        `M2M100Translator`
        `MarianMTTranslator`
        `NLLB200Translator`

    The main call for `Translator` is translate. This function translates required texts and save them to a csv file. 
    """
    def __init__(self, data_path, target_language):
        """
        Initialize the Translator class.
        """
        self.target_language = target_language
        self.data_path = os.path.join(data_path, target_language)
        self.csv_path = os.path.join(self.data_path, 'translations.csv')
        os.makedirs(self.data_path, exist_ok=True)
        
        self.log_path = os.path.join(self.data_path, 'logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.log_counter = max(
            [
                int(f[:-3])
                for f in os.listdir(self.log_path)
                if f.endswith('.to')
            ],
            default=0,
        ) + 1
        
        self.logger = logging.getLogger(__name__)
        self.loaded = False
        

    def load(self):
        try:
            self.dataframe = pd.read_csv(self.csv_path, na_values=[]).set_index('from')
        except FileNotFoundError:
            self.dataframe = pd.DataFrame(columns=['from', 'to']).set_index('from')
        self.loaded = True
        self.logger.info(f'Loaded translations: {len(self.dataframe)}')
        return self
    
    
    def save(self) -> None:
        self.dataframe.to_csv(self.csv_path)
        self.logger.info(f'Saved translations: {len(self.dataframe)}')
        
        
    def log_translation(self, text, translation):
        with open(os.path.join(self.log_path, f'{self.log_counter}.from'), 'w') as f:
            f.write(text)
        with open(os.path.join(self.log_path, f'{self.log_counter}.to'), 'w') as f:
            f.write(translation)
        self.log_counter += 1

    
    def translate(self, texts, graceful=1, save=False):
        """
        `graceful`:
          - 0 - Exceptions will stop the translation
          - 1 - Exceptions will NOT stop the translation, but they will be logged
          - 2 - Exceptions will be ignored
        """
        
        if not self.loaded:
            raise RuntimeError
            
        not_translated = set(texts) - set(self.dataframe.index)
        if not_translated:
            translations = self.create_translations(not_translated, graceful, save)
            self.dataframe = pd.concat([self.dataframe, translations])

            if save:
                self.save()
            
        translations = {
            text: self.dataframe.loc[text].to
            for text in texts
            if text in self.dataframe.index
        }
        
        if (diff := len(set(texts)) - len(translations)) > 0:
            self.logger.info(f'Missing translations: {diff}')
        
        return translations

        
    def create_translations(self, texts, graceful=1, save=False):
        translations = pd.DataFrame(columns=['from', 'to']).set_index('from')

        for text in tqdm.tqdm(texts):
            try:
                translation = self._call_translation(text)
                
            except Exception as e:
                if graceful == 0:
                    if save:
                        self.dataframe = pd.concat([self.dataframe, translations])     
                        self.save()
                    raise e
                if graceful == 1:
                    self.logger.warning(f'The following exception has occured during the translation of: "{text}"\n\n{traceback.format_exc()}')
                continue
            
            self.log_translation(text, translation)
            translations.loc[text] = translation
            
            
        self.logger.info(f'New translations: {len(translations)}')
        
        return translations

    
    def _call_translation(self, text):
        raise NotImplementedError
        