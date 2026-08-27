using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path = @__DIR__
include(path * "/multitask2.jl")

# ============================================================
# Chargement des données
# ============================================================

Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26nobcc.xyz")
Xtrain10, Ytrain10, ltrain10 = read_data("/Datasets/train10nobcc.xyz")
Xtest, Ytest, lt = read_data("/Datasets/bcc_dft_26.xyz")
Xadd, Yadd, ladd = read_data("/Datasets/bcc_dft_10.xyz")

X = vcat(Xtrain26, Xtrain10, Xtest, Xadd)
Y = vcat(Ytrain26, Ytrain10, Ytest, Yadd)

# ============================================================
# Sélection des jeux d'entraînement / test
# ============================================================

idx = shuffle(1:ltrain26)

train_features = idx[1:(3 * ltrain26) ÷ 4]
train_features_multitask = ltrain26 .+ train_features

test_features = ltrain26 + ltrain10 + 1:ltrain26 + ltrain10 + lt
add_features = ltrain26 + ltrain10 + lt + 9:ltrain26 + ltrain10 + lt + 11

train1 = (e = (p = train_features, s = []), f = (p = [], s = []), v = (p = [], s = []))
train2 = (e = (p = train_features, s = []), f = (p = [], s = []), v = (p = [], s = []))
train3 = (e = (p = train_features, s = [train_features_multitask]), f = (p = [], s = [[]]), v = (p = [], s = [[]]))

test_e = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []))
test_e3 = (e = (p = test_features, s = [[]]), f = (p = [], s = [[]]), v = (p = [], s = [[]]))

add2 = (e = (p = add_features, s = []), f = (p = [], s = []), v = (p = [], s = []))
add3 = (e = (p = [], s = [add_features]), f = (p = [], s = [[]]), v = (p = [], s = [[]]))

# ============================================================
# Paramètres de régression
# ============================================================

ζ = 4
nS = 1
normalisation = true

# ============================================================
# Entraînement des trois modèles
# ============================================================

training_points, mean, std = select_observations(X, Y, train1, true)

K1, kept_lines1, data1, σ²1, ϱ1, η1, mean1, std1 = train_model(X, Y, train1, 0; estimator = "nll", ζ = ζ, normalisation = normalisation)
K2, kept_lines2, data2, σ²2, ϱ2, η2, mean2, std2 = train_model(X, Y, train2, 0; estimator = "nll", ζ = ζ, normalisation = normalisation)
K3, kept_lines3, data3, σ²3, ϱ3, η3, mean3, std3 = train_model(X, Y, train3, 1; estimator = "nll", ζ = ζ, normalisation = normalisation)

# ============================================================
# Prédictions
# ============================================================

m1, S1, K1 = multitask(X, train1, test_e, K1, kept_lines1, data1, σ²1, ϱ1, η1, mean1, std1; ζ = ζ, normalisation = true, denorm = false)
m2, S2, K2 = addData(X, Y, train2, kept_lines2, test_e, add2, 0, K2, data2, σ²2, ϱ2, η2, mean2, std2; ζ = ζ, normalisation = true, denorm = false)
m3, S3, K3 = addData(X, Y, train3, kept_lines3, test_e3, add3, 1, K3, data3, σ²3, ϱ3, η3, mean3, std3; ζ = ζ, normalisation = true, denorm = false)

V1 = sqrt.(max.(diag(S1), 0.0))
V2 = sqrt.(max.(diag(S2), 0.0))
V3 = sqrt.(max.(diag(S3), 0.0))

n = length(test_features)

E_pred_1 = m1[1:n] .* std
E_pred_2 = m2[1:n] .* std
E_pred_3 = m3[1:n] .* std

std_E_1 = V1[1:n] .* std
std_E_2 = V2[1:n] .* std
std_E_3 = V3[1:n] .* std

truth, _, _ = select_observations(X, Y, test_e, true, mean, std)
truth = truth .* std

ε_1 = r2(E_pred_1, truth)
ε_2 = r2(E_pred_2, truth)
ε_3 = r2(E_pred_3, truth)

println("--- Comparaison des performances ---")
println("R² Base           = $(round(ε_1, digits=4))")
println("R² AddData        = $(round(ε_2, digits=4))")
println("R² AddData+Multi  = $(round(ε_3, digits=4))")

# ============================================================
# Dashboard comparatif
# ============================================================

names   = ["Base", "AddData", "AddData + Multitask"]
preds   = [E_pred_1, E_pred_2, E_pred_3]
stds    = [std_E_1, std_E_2, std_E_3]
scores  = [ε_1, ε_2, ε_3]
colors  = [:royalblue, :seagreen, :darkorange]

volume = vcat((Y[k].volume for k in test_features)...)
perm = sortperm(volume)
vol_sorted = volume[perm]

fig = Figure(resolution = (1500, 850), fontsize = 16)

ax_parity = [Axis(fig[1, i],
                   title = "$(names[i])  (R² = $(round(scores[i], digits=3)))",
                   xlabel = "Vrai (eV)",
                   ylabel = i == 1 ? "Prédit (eV)" : "") for i in 1:3]

ax_ev = [Axis(fig[2, i],
              title = "Courbe E–V — $(names[i])",
              xlabel = "Volume (Å³)",
              ylabel = i == 1 ? "Énergie (eV)" : "") for i in 1:3]

for i in 1:3
    lines!(ax_parity[i], truth, truth, color = :black, linestyle = :dash)
    errorbars!(ax_parity[i], truth, preds[i], stds[i], color = (:gray, 0.3))
    scatter!(ax_parity[i], truth, preds[i], color = (colors[i], 0.7), markersize = 8)

    scatter!(ax_ev[i], vol_sorted, truth[perm], color = :black, markersize = 6, label = "DFT")
    band!(ax_ev[i], vol_sorted, preds[i][perm] .- stds[i][perm], preds[i][perm] .+ stds[i][perm],
          color = (colors[i], 0.2))
    lines!(ax_ev[i], vol_sorted, preds[i][perm], color = colors[i], linewidth = 2, label = names[i])
    axislegend(ax_ev[i], position = :cb)
end

linkyaxes!(ax_parity...)
linkxaxes!(ax_parity...)
linkyaxes!(ax_ev...)
linkxaxes!(ax_ev...)

Label(fig[0, :], "Comparaison des trois modèles", fontsize = 22, font = :bold)

display(fig)
save("comparaison_trois_modeles.png", fig)


plot_predictions_pro(E_pred_3, truth, std_E_3)

