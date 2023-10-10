from gender_heuristics.heuristics import *


heuristic_sam = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'сам')
heuristic_nisam = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'нисам')
heuristic_bikh = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'бих')

heuristic_bikh_bio_bila = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'бих', ('био', 'била') )
heuristic_sam_bio_bila = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'сам', ('био', 'била') )


def heuristic_sam_sama(translation, tokens):
    """
    For male we also have the `adverb` condition because `sam` also means `did` 
    """

    if token_by_attribute(tokens, 'text', 'сама', lower=True):
        return 'female'

    if (token := token_by_attribute(tokens, 'text', 'сам', lower=True)) and token.get('upos', '') == 'ADV':
        return 'male'


sr_heuristics = [heuristic_sam, heuristic_nisam, heuristic_bikh, heuristic_bikh_bio_bila, heuristic_sam_bio_bila, heuristic_sam_sama]