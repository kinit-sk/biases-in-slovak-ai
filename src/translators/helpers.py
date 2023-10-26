import logging
import re

from translators.amazon_translate import AmazonTranslate
from translators.deepl import DeepL
from translators.google_translate import GoogleTranslate
from translators.chatgpt import ChatGPT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


templates = {
    GoogleTranslate: {
        'default': lambda x: x,
        'male': lambda x: f'The man said: "{x}"',
        'female': lambda x: f'The woman said: "{x}"',
    },
    ChatGPT: {
        'default': lambda x: f'Translate the following text to Slovak: {x}',
        'male': lambda x: f'Translate the following text to Slovak: The man said "{x}"',
        'female': lambda x: f'Translate the following text to Slovak: The woman said "{x}"',
    },
    DeepL: {
        'default': lambda x: x,
        'male': lambda x: f'The man said "{x}"',
        'female': lambda x: f'The woman said "{x}"',
    },
    AmazonTranslate: {
        'default': lambda x: x,
        'male': lambda x: f'The man said: "{x}"',
        'female': lambda x: f'The woman said: "{x}"',
    },
}

patterns = {
    GoogleTranslate: {
        'male': [
            re.compile(r'Muž povedal: "(?P<translation>.*)"'),
            re.compile(r'Muž povedal: „(?P<translation>.*)“'),
            re.compile(r'Muž povedal: (?P<translation>.*)'),
        ],
        'female': [
            re.compile(r'Žena povedala: "(?P<translation>.*)"'),
            re.compile(r'Žena povedala: „(?P<translation>.*)“'),
            re.compile(r'Žena povedala: (?P<translation>.*)'),
        ],
    },
    ChatGPT: {
        'male': [
            re.compile(r'Muž povedal[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Muz povedal[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Muž sa opýtal[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Muž povedal[:,]? „(?P<translation>.*)“[\.]?'),
            re.compile(r'Muž povedal[:,]? „(?P<translation>.*)"[\.]?'),
        ],
        'female': [
            re.compile(r'Žena povedala[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Žena hovorila[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Žena sa opýtala[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'žena povedala[:,]? "(?P<translation>.*)"[\.]?'),
            re.compile(r'Žena povedala[:,]? „(?P<translation>.*)“[\.]?'),
            re.compile(r'Žena povedala[:,]? „(?P<translation>.*)"[\.]?'),
            re.compile(r'Žena povedala[:,]? “(?P<translation>.*)”[\.]?'),
            re.compile(r'Žena povedala[:,]? ,,(?P<translation>.*)"[\.]?'),
        ],
    },
    DeepL: {
        'male': [
            re.compile(r'Muž povedal: "(?P<translation>.*)"'),
            re.compile(r'Ten(to)? (muž )?povedal: "(?P<translation>.*)"'),
            re.compile(r'Muž sa opýtal: "(?P<translation>.*)"'),
            re.compile(r'Muž uviedol: "(?P<translation>.*)"'),
            re.compile(r'Povedal: "(?P<translation>.*)"'),
            re.compile(r'"(?P<translation>.*)"'),
        ],
        'female': [
            re.compile(r'Žena povedala: "(?P<translation>.*)"'),
            re.compile(r'Tá(to)? žena povedala: "(?P<translation>.*)"'),
            re.compile(r'Žena sa [os]?pýtala: "(?P<translation>.*)"'),
            re.compile(r'Žena uviedla: "(?P<translation>.*)"'),
            re.compile(r'"(?P<translation>.*)" opýtala sa žena.'),
            re.compile(r'"(?P<translation>.*)"'),
        ],
    },
    AmazonTranslate: {
        'male': [
            re.compile(r'Muž (od)?povedal: „(?P<translation>.*)“'),
        ],
        'female': [
            re.compile(r'Žena (od)?povedala: „(?P<translation>.*)“'),
        ],
    },
}


def gender_translate(sentence, translator, gender):
    template = templates[type(translator)][gender]
    sentence = template(sentence)
    try:
        translation = translator.dataframe.loc[sentence]['to'].strip()
    except KeyError:
        return None
    if gender in ('male', 'female'):
        for pattern in patterns[type(translator)][gender]:
            if (match := re.fullmatch(pattern, translation)):
                translation = match.group('translation')
                break
        else:
            logger.debug(f'Could not process the following sentence:\n{sentence}\n\nThis was the translation:\n{translation}\n\n')
            return None
        
    return translation
