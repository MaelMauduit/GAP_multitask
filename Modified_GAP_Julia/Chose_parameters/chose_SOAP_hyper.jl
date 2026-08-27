using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie
using Random
using Printf

path = @__DIR__
include(path*"/tools.jl")

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

include(path*"/multitask2.jl")

ζ             = 4
nS            = 0      # Nombre de tâches secondaires
normalisation = true    # Normalisation des sorties

# --- Lecture initiale : sert uniquement à fixer le split train/test et les
#     stats de normalisation des sorties Y (qui ne dépendent pas de r_cut/n_max/l_max) ---
X0, Y0, l = read_data("/Datasets/train26.xyz", r_cut=5, n_max=6, l_max=6, sigma=0.5)

max_train_size   = (3 * l) ÷ 4
full_train_order = fps(X[1:l], max_train_size)
test_features    = setdiff(1:l, full_train_order)

train        = (e = (p = full_train_order, s = []), f = (p = full_train_order, s = []), v = (p = [], s = []))
test_e_and_f = (e = (p = test_features, s = []),   f = (p = test_features, s = []),   v = (p = [], s = []))
test_e       = (e = (p = test_features, s = []),   f = (p = [], s = []),               v = (p = [], s = []))
test_f       = (e = (p = [], s = []),               f = (p = test_features, s = []),   v = (p = [], s = []))

_, mean_ref, std_ref = select_observations(X0, Y0, train, true)
truth, _, _          = select_observations(X0, Y0, test_e, true, mean_ref, std_ref)
truth                = truth .* std_ref
forces, _, _         = select_observations(X0, Y0, test_f, false)

numbers_of_atoms = [size(X0[i][3])[1] for i in test_features]

# --- Étude de convergence sur les hyperparamètres SOAP ---
rs = [8.0, 10.0, 15.0]   
ns = [8, 10, 12]
ls = [6, 8, 10]

@assert length(rs) == length(ns) == length(ls) "rs, ns et ls doivent avoir la même longueur pour le zip"

res = NamedTuple[]

for (r, n, lm) in Iterators.product(rs, ns, ls)
    Xi, Yi, li = read_data("/Datasets/train26.xyz", r_cut=r, n_max=n, l_max=lm, sigma=0.5)
    @assert li == l "Le nombre de configurations a changé avec ces hyperparamètres, vérifie train26.xyz"

    m, S, _ = multitask(Xi, Yi, train, test_e_and_f, nS;
                              ζ = ζ, normalisation = normalisation, denorm = false)

    E_pred = m[1:length(test_features)] .* std_ref
    F_pred = m[length(test_features)+1:end] .* std_ref

    r2_E = r2(E_pred, truth)
    r2_F = r2(F_pred, forces)

    push!(res, (r_cut = r, n_max = n, l_max = lm, r2_E = r2_E, r2_F = r2_F))

    @printf("r_cut=%5.1f  n_max=%3d  l_max=%3d  ->  R²(E) = %.4f   R²(F) = %.4f\n",
            r, n, lm, r2_E, r2_F)
end

println("\n=== Résumé ===")
for row in res
    @printf("r_cut=%5.1f  n_max=%3d  l_max=%3d | R²(E)=%.4f  R²(F)=%.4f\n",
            row.r_cut, row.n_max, row.l_max, row.r2_E, row.r2_F)
end

# --- Plot pour le rapport : heatmap n_max × l_max, une colonne par r_cut, ligne E / ligne F ---
fig = Figure(size = (1000, 700))

all_E = [row.r2_E for row in res]
all_F = [row.r2_F for row in res]
range_E = (minimum(all_E), maximum(all_E))
range_F = (minimum(all_F), maximum(all_F))

for (i, r) in enumerate(rs)
    subres = filter(row -> row.r_cut == r, res)
    grid_E = [only(filter(row -> row.n_max==n && row.l_max==lm, subres)).r2_E for n in ns, lm in ls]
    grid_F = [only(filter(row -> row.n_max==n && row.l_max==lm, subres)).r2_F for n in ns, lm in ls]

    axE = Axis(fig[1, i], title = "R²(E) — r_cut=$r", xlabel = "n_max", ylabel = "l_max",
               xticks = (1:length(ns), string.(ns)), yticks = (1:length(ls), string.(ls)))
    heatmap!(axE, grid_E, colorrange = range_E)

    axF = Axis(fig[2, i], title = "R²(F) — r_cut=$r", xlabel = "n_max", ylabel = "l_max",
               xticks = (1:length(ns), string.(ns)), yticks = (1:length(ls), string.(ls)))
    heatmap!(axF, grid_F, colorrange = range_F)
end

Colorbar(fig[1, length(rs)+1], colorrange = range_E, label = "R²(E)")
Colorbar(fig[2, length(rs)+1], colorrange = range_F, label = "R²(F)")

save(path*"/convergence_R2.png", fig)
fig  