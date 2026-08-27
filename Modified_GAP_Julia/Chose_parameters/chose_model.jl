using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie
using Random

path = @__DIR__
include(path*"/tools.jl")
include(path*"/quip_descriptors.jl")

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

module Opti1
    path = @__DIR__
    include(path*"/multitask.jl")
end

module Opti2
    path = @__DIR__
    include(path*"/multitask2.jl")
end

X, Y, l = read_data("/Datasets/train26.xyz")

ζ             = 2      # Degré du noyau polynomial
nS            = 0      # Nombre de tâches secondaires
normalisation = true    # Normalisation des sorties

max_train_size   = (3 * l) ÷ 4
full_train_order = fps(X[1:l], max_train_size)   # ordre FPS : du + représentatif au + périphérique
test_features    = setdiff(1:l, full_train_order)

test_e_and_f = (e = (p = test_features, s = []), f = (p = test_features, s = []), v = (p = [], s = []))
test_e       = (e = (p = test_features, s = []), f = (p = [], s = []),           v = (p = [], s = []))
test_f       = (e = (p = [], s = []),           f = (p = test_features, s = []), v = (p = [], s = []))

function build_train(train_idx)
    return (e = (p = union(train_idx), s = []), v = (p = [], s = []), f = (p = train_idx, s = []))
end

train_sizes = unique(round.(Int, range(0.1, 1.0, length = 6) .* max_train_size))
train_sizes = train_sizes[train_sizes .> 0]

println("Tailles d'entraînement étudiées : ", train_sizes)

_, mean_ref, std_ref = Opti1.select_observations(X, Y, build_train(full_train_order), true)
truth, _, _          = Opti1.select_observations(X, Y, test_e, true, mean_ref, std_ref)
truth                = truth .* std_ref
forces, _, _         = Opti1.select_observations(X, Y, test_f, false)

numbers_of_atoms = [size(X[i][3])[1] for i in test_features]

# ============================================================
# Boucle d'étude : pour chaque taille, pour chaque optimiseur
# ============================================================

results = (
    size        = Int[],
    r2_energy_1 = Float64[],
    r2_force_1  = Float64[],
    r2_energy_2 = Float64[],
    r2_force_2  = Float64[],
)

for n in train_sizes

    train_idx = full_train_order[1:n]
    train     = build_train(train_idx)

    println("---- Taille d'entraînement : $n / $max_train_size ----")

    # ---- Optim 1 ----
    r2_e1, r2_f1 = try
        m1, S1, _ = Opti1.multitask(X, Y, train, test_e_and_f, nS;
                                     ζ = ζ, normalisation = normalisation, denorm = false)
        E_pred1 = m1[1:length(test_features)] .* std_ref
        F_pred1 = m1[length(test_features)+1:end] .* std_ref
        (Opti1.r2(E_pred1, truth), Opti1.r2(F_pred1, forces))
    catch e
        @warn "Optim 1 a échoué pour n=$n" exception = e
        (NaN, NaN)
    end

    # ---- Optim 2 ----
    r2_e2, r2_f2 = try
        m2, S2, _ = Opti2.multitask(X, Y, train, test_e_and_f, nS;
                                      ζ = ζ, normalisation = normalisation, denorm = false)
        E_pred2 = m2[1:length(test_features)] .* std_ref
        F_pred2 = m2[length(test_features)+1:end] .* std_ref
        (Opti1.r2(E_pred2, truth), Opti1.r2(F_pred2, forces))
    catch e
        @warn "Optim 2 a échoué pour n=$n" exception = e
        (NaN, NaN)
    end

    println("  Optim 1 -> R²(E) = $(round(r2_e1, digits=4))   R²(F) = $(round(r2_f1, digits=4))")
    println("  Optim 2 -> R²(E) = $(round(r2_e2, digits=4))   R²(F) = $(round(r2_f2, digits=4))")

    push!(results.size, n)
    push!(results.r2_energy_1, r2_e1)
    push!(results.r2_force_1,  r2_f1)
    push!(results.r2_energy_2, r2_e2)
    push!(results.r2_force_2,  r2_f2)
end

# ============================================================
# Plot comparatif : courbes d'apprentissage R² vs taille
# ============================================================

function plot_learning_curves(results; savepath = path*"/learning_curves_optim1_vs_optim2.png")

    set_theme!(theme_minimal())
    fig = Figure(size = (1150, 560), fontsize = 16, figure_padding = 24)

    ax1 = Axis(fig[1, 1],
        title  = "R² Énergie vs taille d'entraînement",
        xlabel = "Nombre de structures d'entraînement",
        ylabel = "R²")
    lines!(ax1,   results.size, results.r2_energy_1, color = :dodgerblue, linewidth = 2)
    scatter!(ax1, results.size, results.r2_energy_1, color = :dodgerblue, markersize = 10,
             label = "Optim 1")
    lines!(ax1,   results.size, results.r2_energy_2, color = :orangered, linewidth = 2)
    scatter!(ax1, results.size, results.r2_energy_2, color = :orangered, markersize = 10,
             marker = :utriangle, label = "Optim 2")
    axislegend(ax1, position = :rb, framevisible = false)

    ax2 = Axis(fig[1, 2],
        title  = "R² Forces vs taille d'entraînement",
        xlabel = "Nombre de structures d'entraînement",
        ylabel = "R²")
    lines!(ax2,   results.size, results.r2_force_1, color = :dodgerblue, linewidth = 2)
    scatter!(ax2, results.size, results.r2_force_1, color = :dodgerblue, markersize = 10,
             label = "Optim 1")
    lines!(ax2,   results.size, results.r2_force_2, color = :orangered, linewidth = 2)
    scatter!(ax2, results.size, results.r2_force_2, color = :orangered, markersize = 10,
             marker = :utriangle, label = "Optim 2")
    axislegend(ax2, position = :rb, framevisible = false)

    Label(fig[0, :], "Courbes d'apprentissage — Optim 1 vs Optim 2",
          fontsize = 20, font = :bold)

    save(savepath, fig)
    display(fig)
    return fig
end

fig = plot_learning_curves(results)
# save("compare_perf_1_4_hypers.png", fig)
