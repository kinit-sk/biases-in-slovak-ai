from gender_heuristics.heuristics import *


heuristic_jsem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'jsem')
heuristic_nejsem = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'nejsem')
heuristic_bych = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'bych')
heuristic_ja = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'já')

heuristic_rad_rada = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('rád', 'ráda'))
heuristic_nerad_nerada = lambda translation, tokens: heuristic_gendered_pair(translation, tokens,  ('nerad', 'nerada'))
heuristic_sam_sama = lambda translation, tokens: heuristic_gendered_pair(translation, tokens,  ('sám', 'sama'))

heuristic_jsem_byl_byla = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'jsem', ('byl', 'byla') )


cs_heuristics = [heuristic_jsem, heuristic_nejsem, heuristic_bych, heuristic_ja, heuristic_rad_rada, heuristic_nerad_nerada, heuristic_sam_sama, heuristic_jsem_byl_byla]