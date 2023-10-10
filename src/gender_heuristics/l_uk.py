from gender_heuristics.heuristics import *


heuristic_ia = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'я')
heuristic_buv_bula = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('був', 'була'))
heuristic_odin_odna = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('один', 'одна'))

uk_heuristics = [heuristic_ia, heuristic_buv_bula, heuristic_odin_odna]