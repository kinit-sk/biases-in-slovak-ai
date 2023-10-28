import os

import boto3

from translators.translator import Translator


class AmazonTranslate(Translator):
    aws_access_key = open(os.path.join('config', 'aws_access_key')).read()
    aws_secret_key = open(os.path.join('config', 'aws_secret_key')).read()
    region = 'us-west-2'
    dir_path = os.path.join('cache', 'translations', 'amazon_translate')
    
    
    def __init__(self, target_language, enable_api=False):
        super().__init__(target_language)
        self.enable_api = enable_api
        
        if self.enable_api:
            self.client = boto3.client(
                'translate',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )

    
    def _call_translation(self, text):
        """
        Sometimes this function return errors, e.g., texts in Sorani Kurdish cause BadRequest (as of July 2022).
        
        This function could be optimized in the future by sending multiple texts at the same time. According to API documentation, 5K character requests are optimal: https://cloud.google.com/translate/quotas
        """
        
        if not self.enable_api:
            raise RuntimeError(f'This translator object does not have an API access enabled. Use `GoogleTranslate(enable_api=True)` if you wish to access Google API. You tried to translate: "{text}"')      

        response = self.client.translate_text(
            Text=text,
            SourceLanguageCode='en',
            TargetLanguageCode=self.target_language,
        )
        
        return response['TranslatedText']
