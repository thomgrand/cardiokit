import numpy as np

def tanh_waveform(phi : np.ndarray, t : float, k0 : np.ndarray = -85., 
                  k1 : np.ndarray = 30., tau1 : np.ndarray = 1.) -> np.ndarray:
    xi = phi - t
    vm = k0 + (k1 - k0)/2 * (np.tanh(2 * xi / tau1) + 1)        
    return vm

def tanh_waveform_gen(phi : np.ndarray, t : np.ndarray, **wf_kwargs) -> np.ndarray:
    return (tanh_waveform(phi, t_single, **wf_kwargs) for t_single in t)
