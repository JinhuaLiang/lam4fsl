from .MS_CLAPWrapper import CLAPWrapper as MS_ClapWrapper
from .LAION_CLAPWrapper import CLAPWrapper as LAION_ClapWrapper


def clap_backend(model_name: str):
    if model_name == 'ms_clap':
        return MS_ClapWrapper
    elif model_name == 'laion_clap':
        return LAION_ClapWrapper
