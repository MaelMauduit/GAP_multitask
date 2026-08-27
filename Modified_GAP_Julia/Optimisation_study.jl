using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask.jl")
include( path*"/optimise.jl")
include( path*"/tools.jl")

X, Y, l = read_data("/Datasets/train26.xyz")
indx = (e = (p = union(1:l), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx, false)

pourcentage, result, deviation = convergence_analysis(X, Y, l; percentages = 0.1:0.1:1, n_retry = 3, ζ = 2)

fig = plot_convergence(pourcentage, result, deviation)

fig
# save("Hyperparameters_no_X_normalize_zeta4.png", fig)

function convergence_analysis(X, Y, l::Integer;
                               n_retry::Int = 5,
                               percentages = 0.1:0.1:1.0,
                               nS = 0,
                               normalisation::Bool = true,
                               ζ::Int = 2)
 
    pourcentage = collect(percentages)
    n_p = length(pourcentage)
 
    result    = zeros(Float64, n_p)
    deviation = zeros(Float64, n_p)
 
    for (p_idx, pct) in enumerate(pourcentage)
        result_p = Float64[]
        n_train = clamp(round(Int, pct * l), 1, l)  # nb d'indices, jamais 0 ni > l
 
        for _ in 1:n_retry
            idx = shuffle(1:l)
            train_features = idx[1:n_train]
 
            train = (
                e = (p = union(train_features), s = []),
                v = (p = [], s = []),
                f = (p = train_features, s = []),
            )
 
            kept_lines = create_filter_cov(X, train,nS,ζ)
            σ², ϱ, η = Global_optimizer(X, Y, train, kept_lines, nS, normalisation, ζ)
 
            push!(result_p, σ².e.p)
        end
 
        result[p_idx]    = Statistics.mean(result_p)
        deviation[p_idx] = Statistics.std(result_p)
 
        normalisation && println("Pourcentage $(round(pct*100, digits=1))% -> ",
                            "moyenne = $(round(result[p_idx], digits=4)), ",
                            "std = $(round(deviation[p_idx], digits=4))")
    end
 
    return pourcentage, result, deviation
end
 
# =============================================================================
# PLOT (Makie)
# =============================================================================
 

function plot_convergence(pourcentage, result, deviation;
                           titre = "Dependence of σp² on the Percentage of Selected Data (over 202 samples)")
 
    x = pourcentage .* 100
    ymin = result .- deviation
    ymax = result .+ deviation
 
    set_theme!(theme_minimal())
 
    fig = Figure(size = (900, 550), fontsize = 14)
 
    ax = Axis(fig[1, 1];
        title = titre,
        titlesize = 18,
        xlabel = "Training data percentage (%)",
        ylabel = "σp²",
        xlabelsize = 14,
        ylabelsize = 14,
        xgridcolor = (:grey, 0.15),
        ygridcolor = (:grey, 0.15),
    )
 
    # bande d'incertitude (± écart-type)
    band!(ax, x, ymin, ymax; color = (:royalblue, 0.18), label = "± standard deviation")
 
    # barres d'erreur
    errorbars!(ax, x, result, deviation;
        color = (:royalblue, 0.7), whiskerwidth = 8, linewidth = 1.5)
 
    # ligne moyenne
    lines!(ax, x, result; color = :royalblue, linewidth = 3, label = "mean")
 
    # marqueurs
    scatter!(ax, x, result;
        color = :white, strokecolor = :royalblue, strokewidth = 2.5, markersize = 12)
 
    axislegend(ax; position = :rb, framevisible = false)
 
    xlims!(ax, minimum(x) - 5, maximum(x) + 5)
 
    return fig
end



