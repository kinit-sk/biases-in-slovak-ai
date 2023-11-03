from gender_heuristics.heuristics import *


heuristic_ia = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'я')
heuristic_sam_sama = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('сам', 'сама'))

def heuristic_ia_byu_byla(translation, tokens):
    """
    I think this is an incorrect parsing, but sometimes я (ia) and быў/была (byu/byla) have a common head.
    """
    ia_token = token_by_attribute(tokens, 'text', 'я', lower=True)

    if ia_token:

        byu_token = token_by_attribute(tokens, 'text', 'быў', lower=True)
        if byu_token and byu_token['head'] == ia_token['head']:
            return 'male'

        byla_token = token_by_attribute(tokens, 'text', 'была', lower=True)
        if byla_token and byla_token['head'] == ia_token['head']:
            return 'female'    


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


be_heuristics = [heuristic_ia_byu_byla, heuristic_ia, heuristic_sam_sama, heuristic_ia_wrong]