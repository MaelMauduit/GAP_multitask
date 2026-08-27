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

include(path*"/multitask2.jl")

X, Y, l = read_data("/Datasets/train26.xyz")

zetas         = [2, 4, 6, 8]   # Degrés du noyau polynomial à comparer

nS            = 0      # Nombre de tâches secondaires
normalisation = true    # Normalisation des sorties

k_folds = 5             # Nombre de folds pour la cross-validation
Random.seed!(42)        # Reproductibilité du découpage en folds

function build_train(train_idx)
    return (e = (p = union(train_idx), s = []), f = (p = train_idx, s = []), v = (p = [], s = []))
end

function build_test(test_idx)
    e_and_f = (e = (p = test_idx, s = []), f = (p = test_idx, s = []), v = (p = [], s = []))
    e_only  = (e = (p = test_idx, s = []), f = (p = [], s = []),       v = (p = [], s = []))
    f_only  = (e = (p = [], s = []),       f = (p = test_idx, s = []), v = (p = [], s = []))
    return e_and_f, e_only, f_only
end

# ============================================================
# Construction des folds (k-fold CV)
# ============================================================

perm  = randperm(l)
folds = [perm[i:k_folds:end] for i in 1:k_folds]   # répartition entrelacée -> tailles quasi égales

println("Taille des folds : ", length.(folds))

# ============================================================
# Boucle de cross-validation : pour chaque fold, pour chaque zeta
# ============================================================

results = Dict(ζ => (r2_energy = Float64[], r2_force = Float64[]) for ζ in zetas)

for (fold_num, test_idx) in enumerate(folds)

    train_idx = setdiff(1:l, test_idx)
    train     = build_train(train_idx)
    test_e_and_f, test_e, test_f = build_test(test_idx)

    println("---- Fold $fold_num/$k_folds : train=$(length(train_idx)) test=$(length(test_idx)) ----")

    _, mean_ref, std_ref = select_observations(X, Y, train, true)
    truth, _, _          = select_observations(X, Y, test_e, true, mean_ref, std_ref)
    truth                = truth .* std_ref
    forces, _, _         = select_observations(X, Y, test_f, false)

    for ζ in zetas

        r2_e, r2_f = try
            m, S, _ = multitask(X, Y, train, test_e_and_f, nS;
                                 ζ = ζ, normalisation = normalisation, denorm = false)
            E_pred = m[1:length(test_idx)] .* std_ref
            F_pred = m[length(test_idx)+1:end] .* std_ref
            (r2(E_pred, truth), r2(F_pred, forces))
        catch e
            @warn "ζ=$ζ a échoué pour le fold $fold_num" exception = e
            (NaN, NaN)
        end

        println("  ζ=$ζ -> R²(E) = $(round(r2_e, digits=4))   R²(F) = $(round(r2_f, digits=4))")

        push!(results[ζ].r2_energy, r2_e)
        push!(results[ζ].r2_force,  r2_f)
    end
end

# ============================================================
# Résumé statistique
# ============================================================

println("\n==== Résumé de la cross-validation ($k_folds folds) ====")
for ζ in zetas
    e_vals = results[ζ].r2_energy
    f_vals = results[ζ].r2_force
    println("ζ = $ζ  ->  R²(E) = $(round(mean(e_vals), digits=4)) ± $(round(std(e_vals), digits=4))   " *
            "R²(F) = $(round(mean(f_vals), digits=4)) ± $(round(std(f_vals), digits=4))")
end

# ============================================================
# Plot comparatif : boxplots R² par zeta
# ============================================================

function plot_cv_boxplots(results, zetas, k_folds; savepath = path*"/cv_boxplots_zeta_comparison.png")

    set_theme!(theme_minimal())
    fig = Figure(size = (1000, 520), fontsize = 16, figure_padding = 24)

    ax1 = Axis(fig[1, 1],
        title  = "R² Énergie (cross-validation)",
        xlabel = "ζ",
        ylabel = "R²",
        xticks = (1:length(zetas), string.(zetas)))

    ax2 = Axis(fig[1, 2],
        title  = "R² Forces (cross-validation)",
        xlabel = "ζ",
        ylabel = "R²",
        xticks = (1:length(zetas), string.(zetas)))

    for (i, ζ) in enumerate(zetas)
        e_vals = results[ζ].r2_energy
        f_vals = results[ζ].r2_force

        boxplot!(ax1, fill(i, length(e_vals)), e_vals; width = 0.5)
        scatter!(ax1, fill(i, length(e_vals)) .+ (rand(length(e_vals)) .- 0.5) .* 0.15,
                 e_vals, color = (:black, 0.5), markersize = 6)

        boxplot!(ax2, fill(i, length(f_vals)), f_vals; width = 0.5)
        scatter!(ax2, fill(i, length(f_vals)) .+ (rand(length(f_vals)) .- 0.5) .* 0.15,
                 f_vals, color = (:black, 0.5), markersize = 6)
    end

    Label(fig[0, :], "Courbes d'apprentissage — comparaison ζ = $(join(zetas, ", "))",
        fontsize = 20, font = :bold)

    save(savepath, fig)
    display(fig)
    return fig
end

fig = plot_cv_boxplots(results, zetas, k_folds)
# save("compare_perf_zeta_cv.png", fig)