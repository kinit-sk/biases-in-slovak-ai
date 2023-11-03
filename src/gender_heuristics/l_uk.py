from gender_heuristics.heuristics import *


heuristic_ia = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'я')
heuristic_buv_bula = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('був', 'була'))
heuristic_odin_odna = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('один', 'одна'))

def heuristic_ia_buv_bula(translation, tokens):
    """
    I think this is an incorrect parsing, but sometimes я (ia) and быў/была (byu/byla) have a common head.
    """
    ia_token = token_by_attribute(tokens, 'text', 'я', lower=True)

    if ia_token:

        byu_token = token_by_attribute(tokens, 'text', 'був', lower=True)
        if byu_token and byu_token['head'] == ia_token['head']:
            return 'male'

        byla_token = token_by_attribute(tokens, 'text', 'була', lower=True)
        if byla_token and byla_token['head'] == ia_token['head']:
            return 'female'    


uk_heuristics = [heuristic_ia, heuristic_ia_buv_bula, heuristic_odin_odna]