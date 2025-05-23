from collections import namedtuple
from typing import List


Band = namedtuple('Band', ['fc', 'Nr', 'K', 'BW'])


def get_bands_from_conf(fc, Nr, K, BW) -> List[Band]:
    """"
    Gather all the hyperparameters per band into a single data holder
    Each band shall hold the frequency fc, number of antennas, number of subcarriers and BW
    """
    bands = []
    for fc, Nr, K, BW in zip(fc, Nr, K, BW):
        band = Band(fc=fc, Nr=Nr, K=K, BW=BW)
        bands.append(band)
    return bands
