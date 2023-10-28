"""
Class that encapsulates our Google Translation API.


# Usage

- If you want to use this for other projects, create a new service account using the tutorial and change the KEY_PATH. https://cloud.google.com/translate/docs/setup
- Use `enable_google_api=True` when you want to connect to the Google API. This is turned off by default to minimize spending by mistake.


# Supported languages

As of 2022-07-18:
bh af ak sq am ar hy as ay az eu be bn bs bg ca ceb ny zh-CN zh-TW co hr cs da dv nl en eo et ee tl fi fr fy gl lg ka de el gn gu ht ha haw iw hi hmn hu is ig id ga it ja jw kn kk km rw ko kri ku ky lo la lv ln lt lb mk mg ms ml mt mi mr mn my ne nso no or om ps fa pl pt pa qu ro ru sm sa gd sr st sn sd si sk sl so es su sw sv tg ta tt te th ti ts tr tk uk ur ug uz vi cy xh yi yo zu he zh

The full current list can be obtained via:

1. `GoogleTranslate(enable_api=True).client.get_languages()`. This call actually sends request to API so it should not be abused.
2. https://cloud.google.com/translate/docs/languages


# Python API documentation

https://googleapis.dev/python/translation/latest/client.html

"""
import os

# v2 is a basic translation, there is also v3, but it's not needed for our use-cases
# See: https://cloud.google.com/translate/docs/editions
from google.cloud import translate_v2

from translators.translator import Translator


class GoogleTranslate(Translator):
    
    auth_key_path = os.path.join('config', 'useg-395913-05f904c17367.json')
    dir_path = os.path.join('cache', 'translations', 'google_translate')
    
    def __init__(self, target_language, enable_api=False):
        super().__init__(target_language)
        self.enable_api = enable_api
        
        if self.enable_api:
            self.client = translate_v2.Client.from_service_account_json(self.auth_key_path)

    
    def _call_translation(self, text):
        """
        Sometimes this function return errors, e.g., texts in Sorani Kurdish cause BadRequest (as of July 2022).
        
        This function could be optimized in the future by sending multiple texts at the same time. According to API documentation, 5K character requests are optimal: https://cloud.google.com/translate/quotas
        """
        
        if not self.enable_api:
            raise RuntimeError(f'This translator object does not have an API access enabled. Use `GoogleTranslate(enable_api=True)` if you wish to access Google API. You tried to translate: "{text}"')
        
        response = self.client.translate(
            text,
            source_language='en',
            target_language=self.target_language,
            format_='text',  # `text` is needed because default `format_='html'` escapes special characters
        )
        
        return response['translatedText']
