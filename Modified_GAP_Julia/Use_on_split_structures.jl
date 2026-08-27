using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie
using Printf

path = @__DIR__
include(path * "/multitask2.jl")

# ============================================================
# Chargement des jeux de données complets
# ============================================================
# Xbcc, Ybcc, lbcc = read_data("/Datasets/bcc.xyz")
# Xfcc, Yfcc, lfcc = read_data("/Datasets/fcc.xyz")
# Xhcp, Yhcp, lhcp = read_data("/Datasets/hcp.xyz")
# Xdia, Ydia, ldia = read_data("/Datasets/dia.xyz")
# Xtest, Ytest, ltest = read_data("/Datasets/fcc_july_dft_26.xyz")
# Xtestbcc, Ytestbcc, ltestbcc = read_data("/Datasets/bcc_dft_26.xyz")

X = vcat(Xbcc, Xfcc, Xhcp, Xdia, Xtest, Xtestbcc)
Y = vcat(Ybcc, Yfcc, Yhcp, Ydia, Ytest, Ytestbcc)

l = (
    bcc  = (first = 1, last = lbcc),
    fcc  = (first = 1 + lbcc, last = lbcc + lfcc),
    hcp  = (first = 1 + lbcc + lfcc, last = lhcp + lbcc + lfcc),
    dia  = (first = 1 + lbcc + lfcc + lhcp, last = ldia + lbcc + lfcc + lhcp),
    testfcc = (first = 1 + ldia + lbcc + lfcc + lhcp, last = ltest + ldia + lbcc + lfcc + lhcp),
    testbcc = (first = 1 + ldia + lbcc + lfcc + lhcp + ltest, last = ltestbcc + ltest + ldia + lbcc + lfcc + lhcp),
)

# ============================================================
# Extraction des positions atomiques par phase
# ============================================================
Bx = vcat((X[k][3] for k in l.bcc.first:l.bcc.last)...)
Fx = vcat((X[k][3] for k in l.fcc.first:l.fcc.last)...)
Hx = vcat((X[k][3] for k in l.hcp.first:l.hcp.last)...)
Dx = vcat((X[k][3] for k in l.dia.first:l.dia.last)...)
Tx = vcat((X[k][3] for k in l.testfcc.first:l.testfcc.last)...)
Txb = vcat((X[k][3] for k in l.testbcc.first:l.testbcc.last)...)

test_features = union(l.testbcc.first:l.testbcc.last) #, l.testbcc.first:l.testbcc.last)

test_e        = (e = (p = test_features, s = []), f = (p = [], s = []),            v = (p = [], s = []))
test_f        = (e = (p = [], s = []),            f = (p = test_features, s = []), v = (p = [], s = []))
test_e_and_f  = (e = (p = test_features, s = []), f = (p = test_features, s = []), v = (p = [], s = []))
volume = vcat((Y[k].volume for k in test_features)...)

ζ = 4                  # Degré du noyau polynomial
nS = 0                 # Nombre de tâches secondaires
normalisation = true   # Normalisation des sorties

# ============================================================
# Matrice de distances entre phases (calculée, affichée plus tard)
# ============================================================
labels_dist = ["Bcc", "Fcc", "Hcp", "Dia", "Testfcc", "Testbcc"]
L = [Bx, Fx, Hx, Dx, Tx, Txb]
n = length(L)
D = [Mahalanobis_distance(a, b) for a in L, b in L]

# ============================================================
# Entraînement phase par phase + évaluation sur le jeu test DFT
# ============================================>================
phases = (bcc = l.bcc,)#, fcc = l.fcc, hcp = l.hcp, dia = l.dia)

results = NamedTuple[]   # on stocke tout ici, on affichera à la fin

for (name, rng) in pairs(phases)
    train_features = union(rng.first:rng.last)

    # HYPOTHÈSE : entraînement mono-phase, énergies + forces, pas de tâches secondaires (nS = 0)
    train = (
        e = (p = train_features, s = []),
        f = (p = train_features, s = []),
        v = (p = [], s = []),
    )
    training_points, mean, std = select_observations(X, Y, train, true)

    σ² = ( e=( p=9/std, s=[] ), f=( p=9/std, s=[] ), v=( p=9/std, s=[] ))
    η  = ( e=( p=1e-6/std, s=[] ), f=( p=0.01/std, s=[] ), v=( p=0.01/std, s=[] ))
    ϱ  = ( e=[], f=[], v=[] ) 
    hyper = σ², ϱ, η
    m, S, K = multitask(
        X, Y, train, test_e_and_f, 0;
        estimator = "nll",
        ζ = ζ,
        normalisation = normalisation,
        denorm = false,
        hyper = hyper

    )
    V = sqrt.(diag(S))

    truth, _, _  = select_observations(X, Y, test_e, true, mean, std)


    truth = truth .* std
    training_points = training_points .* std

    forces, _, _ = select_observations(X, Y, test_f, false)

    E_pred = m[1:length(test_features)] .* std
    std_E  = V[1:length(test_features)] .* std
    F_pred = m[length(test_features)+1:end] .* std
    std_F  = V[length(test_features)+1:end] .* std

    numbers_of_atoms = [size(X[i][3])[1] for i in test_e_and_f.e.p]
    for s in test_e_and_f.e.s
        append!(numbers_of_atoms, [size(X[i][3])[1] for i in s])
    end

    numbers_of_atoms_train = [size(X[i][3])[1] for i in train.e.p]
    for s in train.e.s
        append!(numbers_of_atoms_train, [size(X[i][3])[1] for i in s])
    end

    y_true = truth ./ numbers_of_atoms
    y_pred = E_pred ./ numbers_of_atoms
    std_E = std_E ./ numbers_of_atoms

    training_points = training_points[1:1:length(train_features)] ./ numbers_of_atoms_train

    ε       = r2(y_pred, y_true)
    εforces = r2(F_pred, forces)

    train_volume = vcat((Y[k].volume for k in train_features)...)

    fig = plot_predictions_pro(y_pred, y_true, std_E, F_pred, forces, std_F; model_name = "GNN-v2")
    fig2 = plot_result(y_pred, std_E, y_true, volume, training_points, train_volume , ε)
    push!(results, (name = name, energy_r2 = ε, force_r2 = εforces, fig = fig, fig2=fig2, K=K, std_E=std_E))
end

# ============================================================
# AFFICHAGE FINAL — tout regroupé et propre
# ============================================================

println("="^50)
println("MATRICE DE DISTANCES ENTRE PHASES")
println("="^50)
@printf("%8s", "")
for lbl in labels_dist
    @printf("%10s", lbl)
end
println()
for i in 1:n
    @printf("%8s", labels_dist[i])
    for j in 1:n
        @printf("%10.4g", D[i, j])
    end
    println()
end

println()
println("="^50)
println("RÉSULTATS D'ENTRAÎNEMENT (test = jeu DFT)")
println("="^50)
@printf("%-8s %12s %12s\n", "Phase", "Energy R²", "Force R²")
println("-"^34)
for r in results
    @printf("%-8s %12.4f %12.4f\n", r.name, r.energy_r2, r.force_r2)
end
println("="^50)
results[1].fig2
results[1].std_E

results[1].K

