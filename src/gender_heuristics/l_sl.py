from gender_heuristics.heuristics import *


heuristic_sem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'sem')
heuristic_nisem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'nisem')
heuristic_bi = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'bi')
heuristic_biti = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'biti')

heuristic_rad = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('rad', 'rada'))
heuristic_sam = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('sam', 'sama'))


sl_heuristics = [heuristic_sem, heuristic_nisem, heuristic_bi, heuristic_biti, heuristic_rad, heuristic_sam]