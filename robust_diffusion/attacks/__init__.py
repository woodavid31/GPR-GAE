from typing import Union

from .prbcd import PRBCD
from .base_attack import Attack
from .prbcd_constrained import LRBCD

ATTACK_TYPE = Union[PRBCD, LRBCD]
SPARSE_ATTACKS = [PRBCD.__name__, LRBCD.__name__]


def create_attack(attack: str, *args, **kwargs) -> Attack:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    attack : str
        Identifier of the attack
    kwargs
        Containing the hyperparameters

    Returns
    -------
    Union[FGSM, GreedyRBCD, PRBCD]
        The created instance
    """
    if not any([attack.lower() == attack_model.__name__.lower() for attack_model in ATTACK_TYPE.__args__]):
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')

    return globals()[attack](*args, **kwargs)


__all__ = [PRBCD, LRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS]
