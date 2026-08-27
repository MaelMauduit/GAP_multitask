# Avant usage :
# ouvrir julia, créer un environement
# ```REPL
# ]activate .
# ```
# depuis le dossier ou se trouve le code puis ajouter les packages (toujours en mode Pkg)
# ```REPL
# ]add DFTK ACEpotentials AtomsBase etc...
# ```
# il faudra eventuellement refaire
# ```REPL
# ]precompile
# ```
# en cas d'erreur de précompilation

# Usage:  julia --project eos_curve.jl [structure_id] [Ecut] [savedir]
#         (defaults: structure_id = 2, Ecut = 10.0 Ha, savedir=results)

using AtomsBuilder
using SimpleCrystals
using DFTK
using ACEpotentials
using AtomsBase
using PseudoPotentialData
using JLD2
using StaticArrays
using LinearAlgebra
using UnitfulAtomic

# l = 7.3 #Bohr probably #around 5.8 for bcc
# bcc_crystal = FCC(l, :Si, SVector(1,1,1))

# d = 10.26   # ≈ 5.43 Å pour le Si
# bcc_crystal = Diamond(d, :Si, SVector(1,1,1))

# l = 5.8   # ≈ 5.43 Å pour le Si
# bcc_crystal = BCC(l, :Si, SVector(1,1,1))

e = 5.5   # bohr (3.84 Å)
f = 9.1  # bohr (6.34 Å)   
bcc_crystal = HCP(e, f, SVector(1,1,1); atomic_symbol=:Si)

lattice0 = bcc_crystal.lattice.primitive_vectors

pseudos = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Si_elem = ElementPsp(:Si, load_psp(pseudos[:Si]))
atoms = [Si_elem for _ in bcc_crystal.atoms]

pos_cart = hcat([at.position for at in bcc_crystal.atoms]...)
pos_frac = [SVector{3}(lattice0 \ pos_cart[:,i]) for i in 1:size(pos_cart,2)]

Ecut = length(ARGS) >= 2 ? parse(Float64, ARGS[1]) : 10.0 # ici tu peux jouer avec le paramètre Ecut
# kgrid = [8, 8, 5] # discrétisation pour les intégrales en Fourier -- comparer avec rapport de victor

k_spacing = 0.1
kgrid = kgrid_from_maximal_spacing(lattice0, k_spacing)
println("the grid is: ", kgrid)  # -> [N1, N2, N3]

scftol = 1e-6 # paramètre de tolérance pour l'algo itératif--idem voir rapport victor

scale_min    = parse(Float64, ARGS[2])
scale_max    = parse(Float64, ARGS[3])
scale_number = parse(Int, ARGS[4])
scalings = range(scale_min, scale_max; length=scale_number) # ici tu peux jouer avec l'échantillonage et l'ampleur de la déformation

isotrop = parse(Bool, ARGS[6])
if isotrop
    scaling_3D = [(a,a,a) for a in scalings]
else
    scaling_3D = [(a,b,c) for a in scalings for b in scalings for c in scalings if a <= b <= c]
end

volumes = Float64[]
energies = Float64[]
forces = Vector{SVector{3,Float64}}[]
stresses = Matrix{Float64}[]

for s in scaling_3D
    a,b,c = s
    bcc_crystal = HCP(a*e, a*f, SVector(1,1,1); atomic_symbol=:Si)
    lattice = bcc_crystal.lattice.primitive_vectors

    # calcul DFT a proprement parler
    model = model_DFT(lattice, atoms, pos_frac; functionals=PBE(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
    scfres = self_consistent_field(basis; tol=scftol)

    # calcul des forces
    F = compute_forces_cart(scfres)

    # calcul des energies
    push!(volumes, model.unit_cell_volume)
    push!(energies, scfres.energies.total)
    push!(forces, F)

    # calcul du stress
    Σ = compute_stresses_cart(scfres)
    push!(stresses, Σ)


    fmax = maximum(norm, F)
    println("s = $(round.(s; digits=4))   V = $(round(model.unit_cell_volume; digits=3)) bohr^3   " *
            "E = $(round(scfres.energies.total; digits=8)) Ha   " *
            "|F|_max = $(round(fmax; digits=6)) Ha/bohr")
end

# println("\nscalings: ", collect(scaling_3D))
# println("volumes : ", volumes)
# println("energies: ", energies)
# println("forces: ", forces)

all_atoms = [bcc_crystal.atoms[k].atomic_symbol for k in 1:length(bcc_crystal.atoms)]

result = (structure = all_atoms,
    lattice0 = Matrix(lattice0),
    pos_frac = pos_frac,
    scalings=scaling_3D,
    energies=energies,
    forces=forces,
    stresses=stresses,
    volumes=volumes) # a compléter selon les besoins


savedir = length(ARGS) >= 3 ? ARGS[5] : "results"
mkpath(savedir)
@save joinpath(savedir, "results_$(Ecut).jld2") all_atoms lattice0 pos_frac volumes energies forces stresses scaling_3D