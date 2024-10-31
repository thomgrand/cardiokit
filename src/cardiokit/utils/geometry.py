import numpy as np
import pyvista as pv
from typing import Tuple

def upsample_x0(mesh : pv.UnstructuredGrid, points : np.ndarray, x0_vals : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cell_inds = mesh.find_closest_cell(points)
    D_inv = np.linalg.inv(mesh.cell_data["metric"][cell_inds].reshape([-1, 3, 3]))
    cell_point_inds = mesh.cells_dict[pv.cell.CellType.TETRA][cell_inds]
    cell_points = mesh.points[cell_point_inds]
    upwind_dir = cell_points - points[:, np.newaxis]
    upwind_ti = np.sqrt(np.einsum("aex,axy,aey->ae", upwind_dir, D_inv, upwind_dir)) + x0_vals[:, np.newaxis]
    init_ti = np.ones(shape=[mesh.n_points]) * np.inf
    np.minimum.at(init_ti, cell_point_inds, upwind_ti)
    ti_mask = init_ti < np.inf
    x0s_us_inds = np.arange(mesh.n_points)[ti_mask]
    x0s_us_vals = init_ti[ti_mask]
    return x0s_us_inds, x0s_us_vals
