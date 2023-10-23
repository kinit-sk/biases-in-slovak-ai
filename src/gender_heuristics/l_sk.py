from gender_heuristics.heuristics import *


heuristic_som = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'som')
heuristic_byt = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'byť')
heuristic_budem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'budem')

heuristic_rad_rada = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('rád', 'rada'))
heuristic_nerad_nerada = lambda translation, tokens: heuristic_gendered_pair(translation, tokens,  ('nerád', 'nerada'))
heuristic_sam_sama = lambda translation, tokens: heuristic_gendered_pair(translation, tokens,  ('sám', 'sama'))

heuristic_som_bol_bola = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'som', ('bol', 'bola') )


sk_heuristics = [heuristic_som, heuristic_byt, heuristic_budem, heuristic_rad_rada, heuristic_nerad_nerada, heuristic_sam_sama, heuristic_som_bol_bola]