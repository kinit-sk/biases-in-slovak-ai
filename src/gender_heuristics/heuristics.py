def run_heuristics_wrapper(heuristics, lazy=True):
    """
    Helper function to put together language-specific heuristics
    """
    if lazy:
        """
        Return first male or female gender
        """
        def heuristic_f(translation, tokens):
            for h in heuristics:
                if (gender := h(translation, tokens)) is not None:
                    return gender

    else:
        """
        Return all results
        """
        def heuristic_f(translation, tokens):
            return [h(translation, tokens) for h in heuristics]

    return heuristic_f


def token_by_attribute(tokens, attribute, value, lower=False):
    """
    Return a token from `tokens` that has the `attribute` with the given `value`
    Helper function to work with the tokens.
    """
    for token in tokens:
        if attribute in token:

            if lower:
                if token[attribute].lower() == value:
                    return token

            else:
                if token[attribute] == value:
                    return token


def token_gender(token):
    if 'Gender=' in token.get('feats', ''):
        gender_idx = token['feats'].index('Gender=')
        gender = token['feats'][gender_idx+7: gender_idx+10]  # extract Fem, Mas or Neu as gender
        return {'Mas': 'male', 'Fem': 'female', 'Neu': None}[gender]
        
    

def heuristic_gendered_head(translation, tokens, target_word):
    """
    Find the tokens that has the same value as `target_word` and check their heads. If any of the heads are gendered, return that gender.
    """

    for token in tokens:
        if token['text'].lower() == target_word:
            head_token = token_by_attribute(tokens, attribute='id', value=token['head'])
            if head_token and head_token['upos'] in ('VERB', 'AUX', 'ADJ', 'DET') and (gender := token_gender(head_token)):
                    return gender



def heuristic_gendered_pair(translation, tokens, pair):

    male, female = pair

    if any(token.get('text', '').lower() == female for token in tokens):
        return 'female'
        
    if any(token.get('text', '').lower() == male for token in tokens):
        return 'male'


def heuristic_gender_pair_with_target(translation, tokens, target_word, pair):
    male, female = pair
    
    target_token = token_by_attribute(tokens, 'text', target_word, lower=True)
    if target_token is None:
        return None

    female_token = token_by_attribute(tokens, 'text', female, lower=True)
    if female_token and female_token['head'] == target_token['head']:
        return 'female'

    male_token = token_by_attribute(tokens, 'text', male, lower=True)
    if male_token and male_token['head'] == target_token['head']:
        return 'male'



