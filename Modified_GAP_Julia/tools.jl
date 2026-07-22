using Printf
using GLMakie
function read_data(file; r_cut=10.0, n_max=10, l_max=8, sigma=0.5)
    desc = SOAPDescriptor(
        species=["Si"],
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma
    )    
    f = path*file
    configs = ase.read(f, index=":")
    l = length(configs)
    println("The curentrly processing file has len ", l)
    X = [stress_describe( c, desc ) for c in configs ]
    print("Descriptors computed ")
    Y = [ ( e=c.info["dft_energy"],
          f=haskey(c.arrays, "dft_force")  ? reshape(c.arrays["dft_force"],:,1)  : zeros(0,1),
          id=haskey(c.info, "id")  ? [c.info["id"]]  : Int[],
          volume=haskey(c.info, "volume")  ? [c.info["volume"]]  : Float64[],
          v=haskey(c.info,   "dft_virial") ? reshape(c.info["dft_virial"],:,1)   : zeros(0,1) )
    for c in configs ]
    return X,Y,l
end

function plot_result(
    m,
    Var,
    truth,
    volume,
    interpolating_points,
    epsilon;
    hyperparams=nothing,
    savedir=nothing
)

    fig = Figure(size=(900, 600))

    title_str = "Energy–Volume Curve — R² = $(round(epsilon, sigdigits=4))"

    if hyperparams !== nothing
        hp_str = join(
            ["$k = $(round(v, sigdigits=3))" for (k,v) in hyperparams],
            " | "
        )
        title_str *= " — " * hp_str
    end

    ax = Axis(
        fig[1,1],
        xlabel = "Volume (Å³)",
        ylabel = "Energy (eV)",
        title = title_str,
        titlesize = 24,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    pred_color = :dodgerblue3

    if Var !== nothing
        σ = sqrt.(Var)
        band!(
            ax,
            volume,
            m .- σ,
            m .+ σ,
            color = (pred_color, 0.12)
        )
    end

    lines!(ax, volume, truth, color=:forestgreen, linewidth=3, label="DFT")
    scatter!(ax, volume, truth, color=:forestgreen, markersize=8)

    lines!(ax, volume, m, color=pred_color, linewidth=3, linestyle=:dash, label="Prediction")

    scatter!(
        ax,
        volume[interpolating_points],
        truth[interpolating_points],
        color=:darkorange,
        marker=:diamond,
        markersize=14,
        strokecolor=:black,
        strokewidth=1,
        label="Training"
    )

    axislegend(
        ax,
        position=:rt,
        framevisible=true,
        backgroundcolor=(:white, 0.9)
    )

    display(fig)

    if savedir !== nothing
        save(savedir * ".png", fig, px_per_unit=3)
    end

    return fig
end

function pseudo_distance(A,B)
    distance = Inf
    for a in eachrow(A) 
        for b in eachrow(B)
            d = norm(a-b, 2)
            distance = min(distance, d)
        end
    end
    return distance
end

function fps(X, n_select)
    if length(X) <= 5
        println("The number of configurations is less than 5, returning all configurations")
        return 1:length(X)
    end
    if n_select > length(X)
        println("The number of configurations to select is greater than the number of configurations, returning all ", length(X), " configurations")
        return 1:length(X)
    end
    keys = 1:length(X)
    selected = randperm(length(X))[1:5]
    D = vcat((X[k][3] for k in selected)...)
    remaining = setdiff(keys, selected)
    distance_vector = [pseudo_distance(D, X[k][3]) for k in remaining]

    while length(selected) < n_select
        maximum_value, pos = maximum(distance_vector), argmax(distance_vector)
        max_index = remaining[pos]
        println("Maximum distance: ", maximum_value, " selected point ", max_index)
        push!(selected, max_index)
        remaining = deleteat!(remaining, pos)
        D = vcat(D, X[max_index][3])
        distance_vector = deleteat!(distance_vector, pos)
        distance_vector = [min(pseudo_distance(X[max_index][3], X[remaining[k]][3]), distance_vector[k]) for k in 1:length(remaining)]
    end
    println(length(selected), " configurations selected out of ", length(X))
    return selected
end

function Cosine_func(u,v,limit)
    nu = norm(u)
    nv = norm(v)
    if nu < limit 
        println("something weird happened")
        return 2
    end
    if nv < limit
        return -2
    end
    return dot(u,v)/(norm(u)*norm(v))
end

function Cosine_func(u,v,limit)
    nu = norm(u)
    nv = norm(v)
    if nu < limit 
        return 2
    end
    if nv < limit
        return -2
    end
    return dot(u,v)/(norm(u)*norm(v))
end

function filter_cov(K_no_hyper)
    all_norms = [norm(K_no_hyper[i,:]) for i in 1:size(K_no_hyper)[1]]
    limit = 1e-10 * maximum(all_norms)
    double = Int[]
    for i in 1:size(K_no_hyper)[1]
        if i in double 
            continue
        end
        still_good = setdiff(1:size(K_no_hyper)[1], double)
        still_good = setdiff(still_good, 1:i)
        for j in still_good
            if Cosine_func(K_no_hyper[i,:], K_no_hyper[j,:],limit) > 0.999 || Cosine_func(K_no_hyper[i,:], K_no_hyper[j,:],limit) < -0.999
                println("Covariance matrix is ill-conditioned, removing row ", j, " and column ", j)
                push!(double, j)
            end
            if Cosine_func(K_no_hyper[i,:], K_no_hyper[j,:],limit) == 2
                push!(double, i)
                continue
            end
            if Cosine_func(K_no_hyper[i,:], K_no_hyper[j,:],limit) == -2
                push!(double, j)
            end
        end
    end  
    kept = setdiff(1:size(K_no_hyper)[1], double)
    println("Number of configurations kept: ", length(kept), " out of ", size(K_no_hyper)[1])

    return kept
end

function create_filter_cov_raw(X,train,nS)
    # careful, may be an issue with multitask as sigma is ill calibrated. Fix the selection in this case.
    vect_one = ones(nS)
    σ² = (e = (p = 1.0, s = 1. * vect_one), f = (p = 1.0, s = 1. * vect_one), v = (p = 1.0, s = 1.0 * vect_one)) 
    η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
    ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
    K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    kept_lines = filter_cov(K_no_hyper)
    return kept_lines
end

function extract_full_rank_submatrix(K::AbstractMatrix; rtol=1e-8, verbose=true)
    n = size(K)[1]
    
    F = qr(K, ColumnNorm())        # RRQR: pivots ordered by information content
    r_diag = abs.(diag(F.R))
    tol = rtol * r_diag[1]
    r = count(r_diag .> tol)

    kept = sort(F.p[1:r])             # indices to keep, as ROW = COLUMN indices

    if verbose
        println("QR-pivoted rank = ", r, " / ", n, "  (dropped ", n - r, " indices)")
    end

    return kept
end

function create_filter_cov(X,train,nS,ζ)
    # careful, may be an issue with multitask as sigma is ill calibrated. Fix the selection in this case.
    vect_one = ones(nS)
    σ² = (e = (p = 1.0, s = 1. * vect_one), f = (p = 1.0, s = 1. * vect_one), v = (p = 1.0, s = 1.0 * vect_one)) 
    η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
    ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
    K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    kept_lines = extract_full_rank_submatrix(K_no_hyper)
    return kept_lines
end

function plot_eigenvalues(K)
    λK  = eigvals(Symmetric(K))
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="index", ylabel="log10|eigenvalue|",
            title="Eigenvalue spectra")
    scatter!(ax, 1:length(λK),  log10.(abs.(λK)),  label="K")
    axislegend(ax, position=:rb)
    fig
end

function plot_predictions(y_pred, y_true, std, number_of_atoms = nothing)
    @assert length(y_true) == length(y_pred) "y_true and y_pred must have equal length"
    y_true = abs.(y_true)
    y_pred = abs.(y_pred)
    if  !isnothing(number_of_atoms)
        y_true = y_true ./ number_of_atoms
        y_pred = y_pred ./ number_of_atoms
    end

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "True",
        ylabel = "Predicted",
        title = "Predicted vs True")

    lo, hi = extrema(vcat(y_true, y_pred))
    pad = 0.05 * (hi - lo)

    scatter!(ax, y_true, y_pred, color = (:steelblue, 0.6), markersize = 8, label = "predictions")
    lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad],
        color = :black, linestyle = :dash, linewidth = 2, label = "y = x")
    errorbars!(
        ax,
        y_true,
        y_pred .- std,
        y_pred .+ std,
        color = :steelblue,
        linestyle = :dash
    )
    xlims!(ax, lo - pad, hi + pad)
    ylims!(ax, lo - pad, hi + pad)
    axislegend(ax, position = :rb)

    fig
end





# ==============================================================================
# Inverse standard normal CDF (Acklam's algorithm) — no extra dependency needed.
# Used for the calibration/reliability panels and their sampling-noise band.
# ==============================================================================
function norminvcdf(p::Real)
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)

    p_low  = 0.02425
    p_high = 1 - p_low

    if p < p_low
        q = sqrt(-2*log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= p_high
        q = p - 0.5
        r = q*q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2*log(1-p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# Category used consistently for the |z|-based color coding (top-level, pure — no state to mix up).
cat_of(zi::Real) = abs(zi) <= 1 ? 1 : (abs(zi) <= 2 ? 2 : 3)

function _fit_stats(residuals, ytrue)
    mae  = Statistics.mean(abs.(residuals))
    rmse = sqrt(Statistics.mean(residuals .^ 2))
    r2   = 1 - sum(residuals .^ 2) / sum((ytrue .- Statistics.mean(ytrue)) .^ 2)
    return mae, rmse, r2
end

# ==============================================================================
# PropertyDiag — bundles everything needed to diagnose ONE property (energy or
# forces) into a single immutable object. This is the key robustness change:
# ytrue/ypred/sigma/residuals/z/idx1/idx2/idx3 for a given property never
# travel around as loose, independently-orderable function arguments, so a
# mix-up like "forces' index mask applied to energy's array" becomes
# structurally impossible rather than just "shouldn't happen if every call
# site is written correctly".
# ==============================================================================
struct PropertyDiag
    label::String
    ytrue::Vector{Float64}
    ypred::Vector{Float64}
    sigma::Vector{Float64}
    resid::Vector{Float64}
    z::Vector{Float64}
    xr::Vector{Float64}
    mae::Float64
    rmse::Float64
    r2::Float64
end

function PropertyDiag(label::String, ytrue, ypred, sigma)
    ytrue = Float64.(ytrue)
    ypred = Float64.(ypred)
    sigma = max.(abs.(Float64.(sigma)), 1e-12)   # guard against zero/negative sigma

    n = length(ytrue)
    length(ypred) == n || throw(DimensionMismatch(
        "$label: ytrue has $n elements but ypred has $(length(ypred))"))
    length(sigma) == n || throw(DimensionMismatch(
        "$label: ytrue has $n elements but sigma has $(length(sigma))"))

    resid = ypred .- ytrue
    z     = resid ./ sigma
    mae, rmse, r2 = _fit_stats(resid, ytrue)

    lo, hi = extrema(vcat(ytrue, ypred))
    pad = 0.05 * (hi - lo)
    xr  = [lo - pad, hi + pad]

    return PropertyDiag(label, ytrue, ypred, sigma, resid, z, xr, mae, rmse, r2)
end

Base.length(pd::PropertyDiag) = length(pd.ytrue)

# ==============================================================================
# Draws all three panels (parity, residuals vs true, calibration) for ONE
# property into column `col` of `fig`. idx1/idx2/idx3
# are computed and consumed entirely inside this single call — they are never
# returned, never passed as arguments, never stored anywhere the other
# property's panels could reach them.
# ==============================================================================
function add_property_panels!(fig, col::Int, pd::PropertyDiag, palette, unit::String)
    c_accent, c_good, c_mid, c_bad, c_err = palette.c_accent, palette.c_good, palette.c_mid, palette.c_bad, palette.c_err
    n = length(pd)

    cats = cat_of.(pd.z)
    idx1, idx2, idx3 = cats .== 1, cats .== 2, cats .== 3
    @assert length(idx1) == n "internal error: index mask length ($(length(idx1))) != $(pd.label) data length ($n)"

    # ---- Row 1: parity plot -------------------------------------------------
    ax1 = Axis(fig[1, col], xlabel = "True ($unit)", ylabel = "Predicted ($unit)",
        title = "Predicted vs. true ($(pd.label))",
        subtitle = @sprintf("MAE = %.4g   RMSE = %.4g   R² = %.4f   (N=%d)", pd.mae, pd.rmse, pd.r2, n),
        aspect = DataAspect())

    errorbars!(ax1, pd.ytrue, pd.ypred, pd.sigma, pd.sigma, color = c_err, linewidth = 1, whiskerwidth = 0)
    scatter!(ax1, pd.ytrue[idx1], pd.ypred[idx1], color = (c_good, 0.65), markersize = 9,
             strokewidth = 0.5, strokecolor = (:white, 0.6), label = "|z| ≤ 1")
    scatter!(ax1, pd.ytrue[idx2], pd.ypred[idx2], color = (c_mid, 0.75), markersize = 9,
             strokewidth = 0.5, strokecolor = (:white, 0.6), label = "1 < |z| ≤ 2")
    scatter!(ax1, pd.ytrue[idx3], pd.ypred[idx3], color = (c_bad, 0.85), markersize = 9,
             strokewidth = 0.5, strokecolor = (:white, 0.6), label = "|z| > 2")
    lines!(ax1, pd.xr, pd.xr, color = c_accent, linestyle = :dash, linewidth = 2)
    xlims!(ax1, pd.xr...); ylims!(ax1, pd.xr...)
    axislegend(ax1, position = :rb, labelsize = 12)

    # ---- Row 2: residuals vs. true ------------------------------------------
    ax3 = Axis(fig[2, col], xlabel = "True ($unit)", ylabel = "Prediction error ($unit)",
        title = "Residuals ($(pd.label))")

    scatter!(ax3, pd.ytrue[idx1], pd.resid[idx1], color = (c_good, 0.65), markersize = 8)
    scatter!(ax3, pd.ytrue[idx2], pd.resid[idx2], color = (c_mid, 0.75), markersize = 8)
    scatter!(ax3, pd.ytrue[idx3], pd.resid[idx3], color = (c_bad, 0.85), markersize = 8)
    hlines!(ax3, [0.0], color = :black, linestyle = :dash, linewidth = 2)
    xlims!(ax3, pd.xr...)

    # ---- Row 3: calibration / reliability, with Wilson sampling-noise band --
    ax4 = Axis(fig[3, col], xlabel = "Nominal confidence level", ylabel = "Empirical coverage",
        title = "Uncertainty calibration ($(pd.label))", aspect = DataAspect())

    levels = collect(union(0.05:0.05:0.95,0.99))
    observed = [Statistics.mean(abs.(pd.z) .<= norminvcdf(0.5 + p/2)) for p in levels]
    calib_err = Statistics.mean(abs.(observed .- levels))
    ax4.subtitle = @sprintf("Mean calibration error = %.3f   (N = %d)", calib_err, n)

    lines!(ax4, [0, 1], [0, 1], color = c_accent, linestyle = :dash, linewidth = 2, label = "Perfect calibration")
    lines!(ax4, levels, observed, color = c_bad, linewidth = 2.5)
    scatter!(ax4, levels, observed, color = c_bad, markersize = 8, label = "Model")
    xlims!(ax4, 0, 1); ylims!(ax4, 0, 1)
    axislegend(ax4, position = :lt, labelsize = 11)

    return nothing
end

# ==============================================================================
# Main entry point
# ==============================================================================
"""
    plot_predictions_pro(e_pred, e_true, std, f_pred, f_true, std_f;
                          model_name="", save_path=nothing)

Six-panel diagnostic figure for a regression model (e.g. an interatomic
potential) that predicts both energies and forces together with a per-sample
predictive standard deviation for each.

For energies and, separately, for forces:
  • Parity plot (predicted vs. true), points colored by |residual|/σ
  • Residuals vs. true value
  • Calibration (reliability) curve: nominal vs. empirical coverage, with a
    95% sampling-noise band (Wilson score interval) so deviations from the
    diagonal can be judged against what a finite sample alone would produce

`std` / `std_f` are the model's predicted per-sample standard deviations for
energies and forces respectively. `e_*` and `f_*` may have different lengths
(e.g. one energy per structure vs. three force components per atom) — each
property is validated and plotted independently.
"""
function plot_predictions_pro(e_pred, e_true, std, f_pred, f_true, std_f;
                               model_name::String = "", save_path::Union{Nothing,String} = nothing)

    energy = PropertyDiag("energy", e_true, e_pred, std)
    forces = PropertyDiag("forces", f_true, f_pred, std_f)

    # -------------------------------------------------------------------------
    # Palette & theme — clean "editorial" look
    # -------------------------------------------------------------------------
    c_bg     = "#FBFBFD"
    c_grid   = "#E7E9EE"
    c_accent = "#264653"   # slate, used for diagonal / reference lines
    c_good   = "#2A9D8F"   # teal    -> |z| <= 1
    c_mid    = "#E9C46A"   # amber   -> 1 < |z| <= 2
    c_bad    = "#E76F51"   # coral   -> |z| > 2
    c_err    = (:gray40, 0.25)
    palette  = (c_bg = c_bg, c_grid = c_grid, c_accent = c_accent,
                c_good = c_good, c_mid = c_mid, c_bad = c_bad, c_err = c_err)

    set_theme!(Theme(
        fontsize = 15,
        font = "TeX Gyre Heros Makie",
        Axis = (
            backgroundcolor = c_bg,
            xgridcolor = c_grid, ygridcolor = c_grid,
            xgridwidth = 1, ygridwidth = 1,
            topspinevisible = false, rightspinevisible = false,
            titlefont = "TeX Gyre Heros Bold Makie",
            titlesize = 16,
            subtitlefont = "TeX Gyre Heros Makie",
            subtitlesize = 13,
            subtitlecolor = (:black, 0.6),
        ),
        Legend = (framevisible = true, backgroundcolor = (:white, 0.8), padding = (8,8,8,8)),
    ))

    fig = Figure(size = (1250, 1500), backgroundcolor = c_bg)

    Label(fig[0, 1:2],
        isempty(model_name) ? "Model performance diagnostics" : "Model performance diagnostics — $model_name",
        fontsize = 22, font = "TeX Gyre Heros Bold Makie")

    add_property_panels!(fig, 1, energy, palette, "Ha/atom")
    add_property_panels!(fig, 2, forces, palette, "Ha/bohr")

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 18)

    if !isnothing(save_path)
        save(save_path, fig, px_per_unit = 3)
    end

    return fig
end