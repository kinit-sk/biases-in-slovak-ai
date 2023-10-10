from gender_heuristics.heuristics import *


heuristic_ia = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'я')
heuristic_byt = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'быть')
heuristic_odin_odna = lambda translation, tokens: heuristic_gendered_pair(translation, tokens, ('один', 'одна'))

heuristic_ia_bil_bila = lambda translation, tokens: heuristic_gender_pair_with_target(translation, tokens, 'я', ('был', 'была') )


ru_heuristics = [heuristic_ia, heuristic_ia_bil_bila, heuristic_byt, heuristic_odin_odna]