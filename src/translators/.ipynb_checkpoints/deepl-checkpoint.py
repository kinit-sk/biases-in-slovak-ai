"""
Class that encapsulates our DeepL API.


# Usage

- You have to put your auth key into `config/deepl.auth`. The key can be found here: https://www.deepl.com/account/summary
- The free and paid APIs have different API URLs. Use the default None `server_url` for the free API and 'https://api.deepl.com' for the paid API.
- Use `enable_google_api=True` when you want to connect to the Google API. This is turned off by default to minimize spending by mistake.


# API and Python documentation (incl. supported languages)

https://pypi.org/project/deepl/
https://www.deepl.com/docs-api/translate-text/translate-text

"""
import os

# See: https://github.com/DeepLcom/deepl-python
import deepl

from translators.translator import Translator


class DeepL(Translator):
    
    auth_key = open(os.path.join('config', 'deepl.auth')).read()
    
    def __init__(self, data_path, enable_api=False, server_url=None):
        super().__init__(data_path)
        self.enable_api = enable_api
        
        if self.enable_api:
            self.client = deepl.Translator(self.auth_key, server_url=server_url)

    
    def _call_translation(self, text):
        
        if not self.enable_api:
            raise RuntimeError(f'This translator object does not have an API access enabled. Use `DeepL(enable_api=True)` if you wish to access DeepL API. You tried to translate: "{text}"')
        
        response = self.client.translate_text(
            text,
            source_lang="EN",
            target_lang="SK",
        )
        
        return response.text
