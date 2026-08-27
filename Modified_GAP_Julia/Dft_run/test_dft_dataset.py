import numpy as np
import h5py
from ase import Atoms
import os

def run_dft(Ecut, scale_min, scale_max, number_of_points, isotrope):
    os.system(
        f"julia --project eos_curve.jl {Ecut} {scale_min} {scale_max} {number_of_points} values_curve_test {isotrope}"
    )

    def parse_forces(raw):
        # raw : array structuré (n_atoms,)
        n_atoms = raw.shape[0]
        out = np.zeros((n_atoms, 3))
        for j in range(n_atoms):
            out[j, 0] = raw[j]["data"]["1"]
            out[j, 1] = raw[j]["data"]["2"]
            out[j, 2] = raw[j]["data"]["3"]
        return out  # (n_atoms, 3)

    path = f"values_curve_test/results_{Ecut}.0.jld2"
    frames = []
    with h5py.File(path, "r") as f:
        # directs
        structure_dft   = list(f["all_atoms"].asstr()[:])
        energies_dft    = f["energies"][:]                          # (4,) float64
        volumes_dft     = f["volumes"][:]                           # (4,) float64

        # scaling : dtype structuré (4,) avec champs 1,2,3
        raw         = f["scaling_3D"][:]
        scalings_dft  = np.array([[r["1"], r["2"], r["3"]] for r in raw])  # (4,3)

        # lattice0 : dtype structuré scalaire avec 9 champs
        raw         = f["lattice0"][()]
        lattice0_dft   = np.array([raw["data"][str(k)] for k in range(1,10)]).reshape(3,3)

        # pos_frac : dtype structuré (2,) avec champs 1,2,3
        raw         = f["pos_frac"][:]
        pos_frac_dft    = np.array([[r["data"]["1"], r["data"]["2"], r["data"]["3"]] for r in raw])  # (2,3)

        # forces : (4,) object -> chaque ref est (2,) structuré
        def parse_forces(raw):
            return np.array([[r["data"]["1"], r["data"]["2"], r["data"]["3"]] for r in raw])  # (n_atoms,3)
        forces_dft      = [parse_forces(f[ref][()]) for ref in f["forces"][:]]   # list de (2,3)
        
        stresses_dft = [f[ref][()] for ref in f["stresses"][:]]  # list de (3,3)

        n = len(energies_dft)
        assert n == len(forces_dft) == len(stresses_dft) == len(scalings_dft) == len(volumes_dft)

        
        for k in range (n):
            scaling = list(scalings_dft[k])
            lattice = lattice0_dft @ np.diag(scaling)

            energy = energies_dft[k]
            forces = forces_dft[k]
            stresses = stresses_dft[k]
            volume = volumes_dft[k]

            atoms = Atoms(structure_dft,
                            scaled_positions = pos_frac_dft,
                            cell = lattice,
                            pbc = True
                            )
            atoms.info["dft_energy"] = energy
            atoms.info["volume"] = volume
            atoms.arrays["dft_force"] = forces 
            atoms.info["dft_virial"] = stresses
            atoms.info["scaling"] = scaling
            frames.append(atoms)

    # write trajectory
    from ase.io import write
    write(f"hcp_report_dft_{Ecut}.xyz", frames)
    print(f"hcp_report_dft_{Ecut}.xyz", "done")