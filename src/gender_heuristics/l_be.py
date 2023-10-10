from gender_heuristics.heuristics import *


heuristic_ia = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'я')
heuristic_byu_byla = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('быў', 'была'))
heuristic_sam_sama = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('сам', 'сама'))


def heuristic_ia_wrong(translation, tokens):
    """
    Sometimes я is parsed incorrectly as a part of the subsequent word
    """
    
    if 'я ' not in translation.lower():
        return None

    for token in tokens:
        if token.get('text', '').lower().startswith('я '):
            if (gender := token_gender(token)) is not None:
                return gender


be_heuristics = [heuristic_ia, heuristic_byu_byla, heuristic_sam_sama, heuristic_ia_wrong]