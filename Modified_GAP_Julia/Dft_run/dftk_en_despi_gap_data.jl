using DFTK
using PseudoPotentialData
using LinearAlgebra
using Unitful
using UnitfulAtomic

Bohr_to_Ang = 0.529177

# -------------------------
# Cellule (Angström -> Bohr)
# Lattice donnée en lignes (convention extxyz)
# -------------------------
lattice_ang = [
5.8813459194305535 0.19660221161540234 0.019428519951673745 
-0.008109532896696537 5.874398870456681 -0.14663042625662723 
-0.0015577012445290223 0.03240485350973149 5.790149135053445
] .* Bohr_to_Ang

# DFTK veut les vecteurs de réseau en colonnes, donc on transpose
lattice = collect(eachcol(permutedims(lattice_ang))) .* u"angstrom"
lattice = [austrip.(v) for v in lattice]   # conversion en Bohr (valeurs nues)
lattice = hcat(lattice...)

# -------------------------
# Positions cartésiennes (Angström) -> fractionnaires
# -------------------------
positions_cart_ang = [
    0.09386321      -0.02612535       0.12239578
    3.20508999       2.87303328       2.87303725 
] .* Bohr_to_Ang
positions_cart_bohr = positions_cart_ang .* austrip(1u"angstrom")

positions_frac = [lattice \ positions_cart_bohr[i, :] for i in 1:2]

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

basis = PlaneWaveBasis(model; Ecut=10, kgrid=kgrid)

# -------------------------
# SCF
# -------------------------
scfres = self_consistent_field(basis; tol=1e-6)
println("Energy = ", scfres.energies.total)

# -------------------------
# Forces
# -------------------------
forces = DFTK.compute_forces_cart(scfres)
println("Forces calculées (Ha/Bohr):")
display(forces)