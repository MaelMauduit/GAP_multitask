# using Statistics
# using MultivariateStats
# using LinearAlgebra, Distributions
# using GLMakie
# using LaTeXStrings
# path  = @__DIR__

# include( path*"/multitask2.jl")





# # ============================================================
# # Loading full available datasets
# # ============================================================

# Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26.xyz") ## Careful to the used dataset
# Xbcc26, Ybcc26, lb26 = read_data("/Datasets/bcc26.xyz") 
# Xtest, Ytest, lt = read_data("/Datasets/bcc_dft_26.xyz")

# X = vcat(Xtrain26, Xbcc26, Xtest)
# Y = vcat(Ytrain26, Ybcc26, Ytest)

# train = (e = (p = 1:ltrain26, s = []), f = (p = 1:ltrain26, s = []), v = (p = [], s = []))

# nS = 0
# vect_one = ones(nS)
# σ² = (e = (p = 1.0, s = 1. * vect_one), f = (p = 1.0, s = 1. * vect_one), v = (p = 1.0, s = 1.0 * vect_one)) 
# η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
# ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
# K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, 4)
# kept_lines = extract_full_rank_submatrix(Y, train, K_no_hyper)

# K1 = K_no_hyper
# K2 = K_no_hyper[kept_lines, kept_lines]
using Makie: wong_colors

set_theme!(theme_latexfonts())

λ1 = sort(real.(eigvals(K1)))
λ2 = sort(real.(eigvals(K2)))
ε = 1e-20
λ1_filtered = [x <= 0 ? ε : x for x in λ1]
λ2_filtered = [x <= 0 ? ε : x for x in λ2]
colors = wong_colors()

fig = Figure(size = (1100, 480), fontsize = 16)

for (i, (λ, name, col)) in enumerate([
        (λ1_filtered, L"K", colors[1]),
        (λ2_filtered, L"K_{QR}", colors[2]),
    ])
    ax = Axis(
        fig[2, i],
        title = L"%$name spectrum",
        subtitle = "N = $(length(λ)) eigenvalues",
        xlabel = "Index i",
        ylabel = "Eigenvalue λᵢ",
        yscale = log10,
        titlesize = 26,
        subtitlesize = 18,
        subtitlecolor = :gray40,
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 18,
        yticklabelsize = 18,
        xgridvisible = false,
        ygridvisible = true,
        ygridstyle = :dash,
        ygridwidth = 0.6,
        ygridcolor = :gray85,
        topspinevisible = false,
        rightspinevisible = false,
        spinewidth = 1.1,
    )

    lines!(ax, eachindex(λ), λ, color = col, linewidth = 2.5)

    hlines!(ax, [ε]; color = :gray50, linestyle = :dash, linewidth = 1.4)
    text!(
        ax, 1, ε;
        text = "threshold = 1e-20",
        align = (:left, :bottom),
        offset = (4, 4),
        fontsize = 16,
        color = :gray50,
    )
end

Label(fig[1, 1:2], "Comparison of the matrix spectrum before and after QR-pivoted selection",
      fontsize = 26, font = :bold)

rowgap!(fig.layout, 1, 20)
colgap!(fig.layout, 1, 40)
rowsize!(fig.layout, 1, Auto())

resize_to_layout!(fig)

save("QR_illustration.png", fig, px_per_unit = 3)
# save("spectre_K1_K2.png", fig, px_per_unit = 4)   # si tu préfères du raster

