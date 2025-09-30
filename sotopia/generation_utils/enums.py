from enum import Enum


class MentalStateGeneration(Enum):
    """
    The algorithm to use for prediction.
    """
    NO_MENTAL_STATE = 0
    ZEROTH_ORDER_MENTAL_STATE = 1
    FIRST_ORDER_MENTAL_STATE = 2
    FIRST_ORDER_MENTAL_STATE_WITH_GROUND_TRUTH = 3


