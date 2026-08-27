using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path = @__DIR__
include(path * "/multitask2.jl")

# ============================================================
# Chargement des données
# ============================================================

Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26nobccCart.xyz")
Xtrain10, Ytrain10, ltrain10 = read_data("/Datasets/train10nobccCart.xyz")
Xtest, Ytest, lt = read_data("/Datasets/bcc_dft_26.xyz")
Xadd, Yadd, ladd = read_data("/Datasets/bcc_dft_10.xyz")

X = vcat(Xtrain26, Xtrain10, Xtest, Xadd)
Y = vcat(Ytrain26, Ytrain10, Ytest, Yadd)

# ============================================================
# Sélection des jeux d'entraînement / test
# ============================================================


train_features = 1:ltrain26
train_features_multitask = ltrain26 .+ train_features

test_features = ltrain26 + ltrain10 + 1:ltrain26 + ltrain10 + lt
volume_test = vcat((Y[k].volume for k in test_features)...)

add_features = ltrain26 + ltrain10 + lt + 9:ltrain26 + ltrain10 + lt + 11
add_features_extremal = union(add_features, [ltrain26 + ltrain10 + lt + 1,ltrain26 + ltrain10 + lt + ladd])


train1 = (e = (p = train_features, s = []), f = (p = train_features, s = []), v = (p = [], s = []))

test_e1 = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []))

add1 = (e = (p = add_features, s = []), f = (p = add_features, s = []), v = (p = [], s = []))

add1ex = (e = (p = add_features_extremal, s = []), f = (p = [], s = []), v = (p = [], s = []))

ζ = 4
normalisation = true


training_points, mean, std = select_observations(X, Y, train1, true)
K1, kept_lines1, data1, σ²1, ϱ1, η1, mean1, std1 = train_model(X, Y, train1, 0; estimator = "nll", ζ = ζ, normalisation = normalisation) #, hyper=(a1,b1,c1))

m = Matrix{Any}(undef, 1, 3)
S = Matrix{Any}(undef, 1, 3)
V = Matrix{Any}(undef, 1, 3)

m[1,1], S[1,1], _ = multitask(X, train1, test_e1, K1, kept_lines1, data1, σ²1, ϱ1, η1, mean1, std1; ζ = ζ, normalisation = true, denorm = true)
m[1,2], S[1,2], _ = addData(X, Y, train1, kept_lines1, test_e1, add1, 0, K1, data1, σ²1, ϱ1, η1, mean1, std1; ζ = ζ, normalisation = true, denorm = true)
m[1,3], S[1,3], _ = addData(X, Y, train1, kept_lines1, test_e1, add1ex, 0, K1, data1, σ²1, ϱ1, η1, mean1, std1; ζ = ζ, normalisation = true, denorm = true)



# --- Données -------------------------------------------------------------
truth, _, _ = select_observations(X, Y, test_e1, false)
nbE = numbers_of_atoms_energy(X, test_e1)
volume = vcat((Y[k].volume for k in test_features)...)

score = fill(NaN, 2, 3)
V = Array{Vector{Float64}}(undef, 2, 3)
for i in 1:1, j in 1:3
    V[i, j] = sqrt.(max.(diag(S[i, j]), 0.0))
    score[i, j] = rmse(m[i, j], truth, nbE.p)
end

add_points, _, _ = select_observations(X, Y, add1, false)
add_volume = vcat((Y[k].volume for k in add_features)...) ./ 2
add_points = add_points[1:3] ./2 

add_pointsex, _, _ = select_observations(X, Y, add1ex, false) 
add_volumeex = vcat((Y[k].volume for k in add_features_extremal)...) ./2
add_pointsex = add_pointsex ./ 2
# tri par volume croissant : indispensable pour que lignes et bande
# d'incertitude soient tracées correctement
order    = sortperm(volume)
volume_s = volume[order] ./ 2
truth_s  = truth[order] ./ 2

# --- Noms des lignes (tâches) et colonnes (méthodes) ------------------------
names_rows = ["Single task", "Multitask"]
names_cols = ["Initial dataset", "Add 3 LF-points", "Add 5 LF-points"]

# --- Figure ------------------------------------------------------------
fig = Figure(size = (1400, 500), fontsize = 20)

axes = Array{Axis}(undef, 1, 3)

for j in 1:3
    ax = Axis(fig[1, j])
    axes[1, j] = ax

    m_s = m[1, j][order] ./ 2
    V_s = V[1, j][order] ./ 2

    # bande d'incertitude à ±1σ autour de la prédiction
    band!(ax, volume_s, m_s .- V_s, m_s .+ V_s;
        color = (:steelblue, 0.25), label = "Uncertainty (±1σ)")

    # prédiction
    lines!(ax, volume_s, m_s; color = :steelblue, linewidth = 2)
    scatter!(ax, volume_s, m_s; color = :steelblue, markersize = 5, label = "Prediction")

    # vérité terrain
    lines!(ax, volume_s, truth_s; color = :black, linewidth = 2, linestyle = :dash)
    scatter!(ax, volume_s, truth_s; color = :black, markersize = 5, label = "Target value")

    if j == 2 
        scatter!(ax, add_volume, add_points;
            color = :darkorange, markersize = 8, label = "Added data")
    end

    if j == 3
        scatter!(ax, add_volumeex, add_pointsex;
            color = :darkorange, markersize = 8, label = "Added data")
    end

    text!(ax, 0.05, 0.95;
        text = "RMSE = $(round(score[1, j], digits = 4))",
        space = :relative, align = (:left, :top), fontsize = 20)

    # format des ticks
    ax.xtickformat = vs -> [@sprintf("%.0f", v) for v in vs]
    ax.ytickformat = vs -> [@sprintf("%.3f", v) for v in vs]

    # labels y uniquement sur la première colonne
    j > 1 && hideydecorations!(ax; grid = false, ticks = false)
end

linkxaxes!(axes...)
linkyaxes!(axes...)

ylims!(axes[1, 1], -4.22, -4.20)

# --- Titres de colonnes ------------------------------------------------
for j in 1:3
    Label(fig[1, j, Top()], names_cols[j];
        fontsize = 26, font = :bold, padding = (0, 0, 10, 0))
end

# --- Titre de ligne ----------------------------------------------------
Label(fig[1, 3, Right()], names_rows[1];
    fontsize = 26, font = :bold,
    rotation = -pi/2, padding = (10, 0, 0, 0))

# --- Labels d'axes communs ---------------------------------------------
Label(fig[1, 1:3, Bottom()], "Volume per atom (Bohr³/atom)";
    fontsize = 24, padding = (0, 0, 0, 35))

Label(fig[1, 1, Left()], "Énergy per atome (Ha/atom)";
    fontsize = 24, rotation = pi/2, padding = (0, 65, 0, 0))

# --- Titre général -----------------------------------------------------
Label(fig[0, 1:3],
    "Comparaison single task vs Multitask with low-fidelity data added";
    fontsize = 26, font = :bold)

# --- Légende unique pour toute la figure -------------------------------
Legend(fig[2, 1:3], axes[1, 2],
    orientation = :horizontal,
    tellheight = true,
    tellwidth = false,
    fontsize = 24)

colgap!(fig.layout, 8)
rowgap!(fig.layout, 12)

save(joinpath(path, "Plot_report", "1_row.png"), fig)