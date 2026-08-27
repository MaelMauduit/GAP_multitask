using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path = @__DIR__
include( path*"/multitask2.jl")

# ============================================================
# Data loading
# ============================================================


function data(name)
    Xdata, Ydata, ldata = read_data("/Datasets/$(name).xyz")
    extract = (e = (p = 1:ldata, s = []), f = (p = [], s = []), v = (p = [], s = []))
    Yd, _, _ = select_observations(Xdata, Ydata, extract, false)
    volume = vcat((Ydata[k].volume for k in 1:ldata)...)
    x = vcat((Xdata[k][3] for k in 1:ldata)...)
    return (v = volume, y = Yd, X=Xdata, x=x)
end


# ============================================================
# Energy-volume curve 
# ============================================================

function plot_energy_volume(volumes::AbstractVector, ys::AbstractVector, labels::AbstractVector;
        colors = [colorant"#0072B2", colorant"#D55E00", colorant"#009E73",
                  colorant"#CC79A7", colorant"#E69F00", colorant"#56B4E9"],
        linestyles = nothing,
        title = "Energy-Volume curve for different Ecut",
        xlabel = "Volume per atom (Bohr³/atom)",
        ylabel = "Energy per atom (Ha/atom)",
        legend_position = :rt,
        size = (900, 600),
        savepath = nothing,
    )

    @assert length(volumes) == length(ys) == length(labels)

    set_theme!(Theme(
        fontsize = 22,
        Axis = (
            backgroundcolor = :white,
            xgridvisible = false,
            ygridvisible = false,
            spinewidth = 1.5,
            xminorticksvisible = true,
            yminorticksvisible = true,
        ),
    ))

    fig = Figure(size = size, figure_padding = 20)

    ax = Axis(fig[1, 1],
        xlabel = xlabel,
        ylabel = ylabel,
        xlabelsize = 24,
        ylabelsize = 24,
        titlesize = 26,
        xticklabelsize = 20,
        yticklabelsize = 20,
        title = title,
    )

    if isnothing(linestyles)
        for (i, (v, y, label)) in enumerate(zip(volumes, ys, labels))
            scatter!(
            ax, v, y,
            color = colors[mod1(i, length(colors))],
            markersize = 7,
            strokewidth = 0.5,
            label = label,
            )
        end

    else
        for (i, (v, y, label)) in enumerate(zip(volumes, ys, labels))
            lines!(ax, v, y,
                color = colors[mod1(i, length(colors))],
                linewidth = 3,
                linestyle = linestyles[mod1(i, length(linestyles))],
                label = label,
            )
        end
    end


    axislegend(ax,
        position = legend_position,
        labelsize = 24,
        patchsize = (40, 20),
        patchlabelgap = 10,
        padding = (14, 14, 10, 10),
        rowgap = 8,
        framevisible = true,
        framecolor = (:black, 0.15),
        backgroundcolor = (:white, 0.85),
    )

    savepath !== nothing && save(savepath, fig)

    return fig
end

Xt26 = data("bcc_dft_26")
Xt18 = data("bcc_dft_18")
Xt10 = data("bcc_dft_10")

## Energy-volume curve highlighting Ecut influence
fig1 = plot_energy_volume(
    [Xt26.v ./ 2, Xt18.v ./ 2, Xt10.v ./ 2],
    [Xt26.y ./ 2, Xt18.y ./ 2, Xt10.y ./ 2],
    [L"E_{cut} = 26", L"E_{cut} = 18", L"E_{cut} = 10"];
    title = L"\text{Energy-Volume curve for different } E_{cut}",
    savepath = joinpath(path, "Plot_report", "energy_vs_volume.png"),
)


bcc = data("bcc_report_dft_26")
fcc = data("fcc_report_dft_26")
dia = data("dia_report_dft_26")
hcp = data("hcp_report_dft_26")

## Energy-volume for different phases  
fig2 = plot_energy_volume(
    [bcc.v ./ 2, fcc.v ./ 4, dia.v ./ 8, hcp.v ./ 2],
    [bcc.y ./ 2, fcc.y ./ 4, dia.y ./ 8, hcp.y ./ 2],
    ["BCC", "FCC", "DIAMOND", "HCP"];
    title = "Energy-Volume curve for different phases",
    savepath = joinpath(path, "Plot_report", "bcc_vs_fcc.png"),
    colors = [colorant"#0072B2",   # blue
            colorant"#CC79A7",   # purple/pink
            colorant"#D55E00",   # vermilion
            colorant"#009E73"],  # green/teal
    linestyles = [:solid, :dash, :dashdot, :dot]
          )

b1 = data("bcc")
f1 = data("fcc")
h1 = data("hcp")
d1 = data("dia")

## Dataset points energy-volume plot
fig2 = plot_energy_volume(
    [b1.v ./2, f1.v ./4, h1.v ./2, d1.v ./2, Xt26.v ./2],
    [b1.y ./2, f1.y ./4, h1.y ./2, d1.y ./2, Xt26.y ./2],
    ["BCC", "FCC", "HCP", "DIAMOND", "TEST"];
    title = "Dataset representation",
    savepath = joinpath(path, "Plot_report", "dataset.png"),
    colors = [colorant"#0072B2",   # blue
            colorant"#CC79A7",   # purple/pink
            colorant"#D55E00",   # vermilion
            colorant"#009E73",
            colorant"#4D4D4D",  # test dataset
            ])


# ============================================================
# PCA plot of the dataset and FPS points selection
# ============================================================

function plot_pca_selected(
    datasets, n_selected;
    labels = vcat(["Group $i" for i in eachindex(datasets)], ["Selected (FPS)"]),
    maxoutdim = 3,
    colors = Makie.wong_colors(),
    base_marker = :circle,
    selected_marker = :xcross,
    savepath = nothing,
)
    nb_datasets = length(datasets)

    for_PCA = hcat([datasets[k].x' for k in 1:nb_datasets]...)
    for_FPS = vcat([datasets[k].X for k in 1:nb_datasets]...)
    selected = Set(fps(for_FPS, n_selected))

    # --- aplatissement : une ligne = un point, avec son groupe et son statut ---
    rows       = Vector{Vector{Float64}}()
    row_group  = Int[]
    row_select = Bool[]

    idx = 0
    for k in 1:nb_datasets
        for d in datasets[k].X
            idx += 1
            is_sel = idx in selected
            mat = d[3]
            for j in 1:size(mat, 1)
                push!(rows, mat[j, :])
                push!(row_group, k)
                push!(row_select, is_sel)
            end
        end
    end

    pca = fit(PCA, for_PCA; maxoutdim = maxoutdim)
    explained = principalvars(pca) ./ sum(principalvars(pca)) .* 100

    # --- projection vectorisée en un seul appel (au lieu de predict() point par point) ---
    all_points = reduce(hcat, rows)      # d × N
    scores = predict(pca, all_points)    # maxoutdim × N
    x, y = scores[1, :], scores[2, :]

    fig = Figure(size = (750, 550), fontsize = 16)
    ax = Axis(
        fig[1, 1],
        xlabel = "PC1 ($(round(explained[1], digits=1))% of variance)",
        ylabel = "PC2 ($(round(explained[2], digits=1))% of variance)",
        xlabelsize = 18, ylabelsize = 18,
        aspect = DataAspect(),
        xgridvisible = false, ygridvisible = false,
    )

    # 1) fond : points non sélectionnés — 1 seul scatter! par groupe
    for k in 1:nb_datasets
        m = (row_group .== k) .& .!row_select
        scatter!(ax, x[m], y[m]; color = colors[k], marker = base_marker,
                  markersize = 10, strokewidth = 0.5, strokecolor = :black, alpha = 0.55)
    end

    # 2) dessus : points sélectionnés — dessinés APRÈS, donc jamais cachés
    for k in 1:nb_datasets
        m = (row_group .== k) .& row_select
        any(m) || continue
        scatter!(ax, x[m], y[m]; color = colors[k], marker = selected_marker,
                  markersize = 13, strokewidth = 1.5, strokecolor = :black, alpha = 1.0)
    end

    hidespines!(ax, :t, :r)

    # --- légende construite à la main : exactement nb_datasets + 1 entrées, jamais de doublon ---
    group_elems = [
        MarkerElement(color = colors[k], marker = base_marker, markersize = 11,
                      strokecolor = :black, strokewidth = 0.5)
        for k in 1:nb_datasets
    ]
    selected_elem = MarkerElement(color = :gray20, marker = selected_marker, markersize = 13,
                                   strokecolor = :black, strokewidth = 1.5)

    legend_elems = copy(group_elems)
    legend_labels = copy(labels[1:nb_datasets])

    if n_selected > 0
        push!(legend_elems, selected_elem)
        push!(legend_labels, "Selected (FPS)")
    end

    Legend(
        fig[1, 1],
        legend_elems,
        legend_labels,
        "Structures";
        framevisible = false,
        tellheight = false,
        tellwidth = false,
        halign = :center,
        valign = :center
    )

    if savepath !== nothing
        save(savepath, fig; px_per_unit = 4)
    end

    return fig
end

## FPS points selection 
plot_pca_selected([b1,f1,h1,d1],30; 
    labels=["BCC", "FCC", "HCP", "Diamond"],
    savepath=joinpath(path, "Plot_report", "PCA1_2.png"),
)



