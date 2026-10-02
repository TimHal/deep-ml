import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:

    norm_exp_sum = sum([math.exp(s - max(scores)) for s in scores])

    return [round(math.exp(x - max(scores))/norm_exp_sum, 4) for x in scores]