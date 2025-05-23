import numpy as np


def array_response_vector(var_array: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * var_array)


def compute_time_options(fc: float, K: int, BW: float, values: np.ndarray,remove_duplicates = 0) -> np.ndarray:
    # create the K frequency bins
    time_basis_vector = np.linspace(fc - BW / 2, fc + BW / 2, K)
    # simulate the phase at each frequency bins
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    # compute the phase response at each subcarrier
    array_response_combination = array_response_vector(combination)
    # might have duplicates depending on the frequency, BW and number of subcarriers
    # so remove the recurring time basis vectors - assume only the first one is valid
    # and the ones after can only cause recovery errors
    if remove_duplicates == 1:
       first_row_duplicates = np.all(np.isclose(array_response_combination, array_response_combination[0]), axis=1)
       if sum(first_row_duplicates) > 1:
           print("some duplicates steering vectors were removed!!")
           dup_row = np.where(first_row_duplicates)[0][1]
           array_response_combination = array_response_combination[:dup_row]
    return array_response_combination


def compute_angle_options(aoa: np.ndarray, values: np.ndarray) -> np.ndarray:
    # simulate the degree at each antenna element
    combination = np.dot(aoa.reshape(-1, 1), values.reshape(1, -1))
    # return the phase at each antenna element in a vector
    return array_response_vector(combination / 2)

def grid_basis_func(band, T_res, aoa_res):
    aoa_grid = np.arange(-np.pi / 2, np.pi / 2, aoa_res * np.pi / 180)
    aoa_basis = compute_angle_options(np.sin(aoa_grid), values=np.arange(band.Nr))
    times_grid = np.arange(0, 0.7*band.K / band.BW, T_res)
    toa_basis = compute_time_options(0, band.K, band.BW, values=times_grid,remove_duplicates = 0)
    if toa_basis.shape[0] < times_grid.shape[0]:
        times_grid = times_grid[:toa_basis.shape[0]]
    return aoa_basis, toa_basis, aoa_grid, times_grid