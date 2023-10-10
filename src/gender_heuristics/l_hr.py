from gender_heuristics.heuristics import *


heuristic_sam = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'sam')
heuristic_nisam = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'nisam')
heuristic_bih = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'bih')

heuristic_sam_bio_bila = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'sam', ('bio', 'bila') )


def heuristic_sam_sama(translation, tokens):
    """
    For male we also have the `adverb` condition because `sam` also means `did` 
    """

    if token_by_attribute(tokens, 'text', 'sama', lower=True):
        return 'female'

    if (token := token_by_attribute(tokens, 'text', 'sam', lower=True)) and token.get('upos', '') == 'ADV':
        return 'male'


hr_heuristics = [heuristic_sam, heuristic_nisam, heuristic_bih, heuristic_sam_bio_bila, heuristic_sam_sama]