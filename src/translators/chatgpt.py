"""
Class that encapsulates ChatGPT API

"""
import os
import time

import openai

from translators.translator import Translator


class ChatGPT(Translator):
    
    auth_key = open(os.path.join('config', 'chatgpt.auth')).read()
    
    def __init__(self, data_path, enable_api=False):
        super().__init__(data_path)
        self.enable_api = enable_api
        
        if self.enable_api:
            openai.api_key = self.auth_key

    
    def _call_translation(self, text):
        
        if not self.enable_api:
            raise RuntimeError(f'This translator object does not have an API access enabled. Use `ChatGPT(enable_api=True)` if you wish to access ChatGPT API. You tried to translate: "{text}"')

        for i in range(5):
            try:
                completion = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': text}]
                )
            except Exception as e:
                if i == 4:
                    raise e
                time.sleep(3)
            else:
                break
        
        return completion.choices[0].message.content
