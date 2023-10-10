from gender_heuristics.heuristics import *


heuristic_jestem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'jestem')
heuristic_byc = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'być')
heuristic_sam_sama = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('sam', 'sama'))


def heuristic_polish_verbs(translation, tokens):
    """
    Polish verbs have a special expanded form in the parse tree.
    """
    
    for token in tokens:
        if 'expanded' in token:
            verb_tokens = token['expanded']
            if 'Person=1' in verb_tokens[-1].get('feats', '') and 'Gender=' in verb_tokens[0].get('feats', ''):
                return token_gender(verb_tokens[0])


def heuristic_gdybym(translation, tokens):
    """
    Similar to gendered_head heuristics, but we need to go through the `expanded` fields
    """
    for token in tokens:
        if token['text'].lower() == 'gdybym':
            head_token = token_by_attribute(tokens, 'id', token['expanded'][0]['head'])
            if (gender := token_gender(head_token)) is not None:
                return gender


def heuristic_czuje(translation, tokens):
    """
    Looking for gendered ADJs and AUXs for thew word _czuję_
    """

    czuje_token = token_by_attribute(tokens, 'text', 'czuję', lower=True)

    if not czuje_token:
        return None   

    for token in tokens:
        if token.get('head', -1) == czuje_token['id'] and token['upos'] in ('AUX', 'ADJ') and (gender := token_gender(token)) is not None:
            return gender


pl_heuristics = [heuristic_polish_verbs, heuristic_jestem, heuristic_byc, heuristic_gdybym, heuristic_czuje, heuristic_sam_sama]