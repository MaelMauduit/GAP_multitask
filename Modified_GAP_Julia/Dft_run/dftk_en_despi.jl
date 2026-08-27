using DFTK
using PseudoPotentialData
using LinearAlgebra

# -------------------------
# Cellule en Bohr (vecteurs en colonnes pour DFTK)
# -------------------------
lattice = [
5.8813459194305535 0.19660221161540234 0.019428519951673745 
-0.008109532896696537 5.874398870456681 -0.14663042625662723 
-0.0015577012445290223 0.03240485350973149 5.790149135053445
]
lattice = permutedims(lattice)

# Positions cartésiennes (Bohr) -> fractionnaires
positions_cart = [
0.09386321      -0.02612535       0.12239578
3.20508999       2.87303328       2.87303725
]
positions_frac = [lattice \ positions_cart[i, :] for i in 1:2]

# -------------------------
# Atomes
# -------------------------
pseudos = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Si = ElementPsp(:Si, load_psp(pseudos[:Si]))
atoms = [Si, Si]

# -------------------------
# Modèle DFT
# -------------------------
model = model_DFT(lattice, atoms, positions_frac; functionals=PBE(), temperature=1e-3)

# -------------------------
# Base
# -------------------------
kgrid = kgrid_from_maximal_spacing(lattice, 0.1)

basis = PlaneWaveBasis(model; Ecut=26, kgrid=kgrid)

# -------------------------
# SCF
# -------------------------
scfres = self_consistent_field(basis; tol=1e-6)
println("Energy = ", scfres.energies.total)
println("dft_energy = -8.423396469899755")
# -------------------------
# Forces
# -------------------------
forces = DFTK.compute_forces(scfres)
println("Forces (Ha/Bohr):")
display(forces)
println("DFT Forces : 0.02922035      -0.02629153      -0.01423315")
