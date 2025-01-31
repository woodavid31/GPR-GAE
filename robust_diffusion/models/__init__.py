from typing import Any, Dict, Union

from robust_diffusion.models.gcn import GCN
from robust_diffusion.models.gprgnn import GPRGNN
from robust_diffusion.models.gat_weighted import GAT
from robust_diffusion.models.MLP import MLP
from robust_diffusion.models.gprgae import GPRGAE


MODEL_TYPE = Union[GCN, GPRGNN, GAT,GPRGAE,MLP]


def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    model: MODEL_TYPE
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == "GPRGNN":
        return GPRGNN(**hyperparams)
    if hyperparams['model'] == "GAT":
        return GAT(**hyperparams)
    if hyperparams['model'] == "GPRGAE":
        return GPRGAE(**hyperparams)
    
    return MLP(**hyperparams)



__all__ = [GCN,
           GPRGNN,
           GAT,
           GPRGAE,
           MLP,
           create_model,
           MODEL_TYPE]
