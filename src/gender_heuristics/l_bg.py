from gender_heuristics.heuristics import *


heuristic_sam = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'съм')
heuristic_bikh = lambda translation, tokens: heuristic_gendered_head(translation, tokens, 'бих')


bg_heuristics = [heuristic_sam, heuristic_bikh]