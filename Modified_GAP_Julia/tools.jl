using Printf
using GLMakie
function read_data(file; r_cut=8.0, n_max=10, l_max=8, sigma=0.5)
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

function r2(m, truth)
    ss_res = sum((m .- truth).^2)
    ss_tot = sum((truth .- Statistics.mean(truth)).^2)
    return 1 - ss_res / ss_tot
end

function rmse(y_true, y_pred, nb_atom=nothing)
    if nb_atom === nothing
        println("⚠️ nb_atom absent → utilisation de 1 pour tous les éléments.")
        nb_atom = ones(length(y_true))
    end

    return sqrt(Statistics.mean(((y_true .- y_pred) ./ nb_atom).^2))
end

rmse_relative(y_true, y_pred) = sqrt(Statistics.mean(((y_true .- y_pred).^2) ./ max.(abs.(y_true), 1e-8).^2))

function plot_result(
    m,
    std,
    truth,
    volume,
    nb_pred,
    training_points,
    training_volumes,
    nb_train,
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
        xlabel = "Volume (Bohr³)",
        ylabel = "Energy (Ha)",
        title = title_str,
        titlesize = 24,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    pred_color = :dodgerblue3
    volume = volume ./ nb_pred
    m = m ./ nb_pred
    truth = truth ./ nb_pred
    if std !== nothing
        std = std ./ nb_pred
        band!(
            ax,
            volume,
            m .- std,
            m .+ std,
            color = (pred_color, 0.12)
        )
    end

    lines!(ax, volume, truth, color=:forestgreen, linewidth=3, label="DFT")
    scatter!(ax, volume, truth, color=:forestgreen, markersize=8)

    lines!(ax, volume, m, color=pred_color, linewidth=3, linestyle=:dash, label="Prediction")
    
    if !isempty(training_points)
        scatter!(
            ax,
            training_volumes ./ nb_train,
            training_points ./ nb_train,
            color=:darkorange,
            marker=:diamond,
            markersize=60,
            strokecolor=:black,
            strokewidth=1,
            label="Training"
        )
    end

    axislegend(
        ax,
        position=:rt,
        framevisible=true,
        backgroundcolor=(:white, 0.9)
    )

    if savedir !== nothing
        save(savedir * ".png", fig, px_per_unit=3)
    end

    return fig
end

function Mahalanobis_distance(R, V)
    μ = Statistics.mean(R, dims=1)
    S = Statistics.cov(R, dims=1)

    Sinv = pinv(S)

    distances = []

    for i in 1:size(V,1)
        x = V[i, :] .- vec(μ)
        d = sqrt(x' * Sinv * x)
        push!(distances, d)
    end

    return Statistics.mean(distances)
end

function angular_distance(a,b)
    ps = dot(a,b)
    if isapprox(ps, 1; atol=1e-8)
        ps = 1.0
    end
    if isapprox(ps, -1; atol=1e-8)
        ps = -1.0
    end
    return acos(ps)
end

function L2_distance(a,b)
    return norm(a-b)
end

function Hausdorff(A, B; distance_metric = angular_distance)
    ab = 0
    ba = 0
    for a in eachrow(A) 
        distance = Inf
        for b in eachrow(B)
            d = distance_metric(a,b)
            distance = min(distance, d)
        end
        ab = max(ab, distance)
    end
    for b in eachrow(B) 
        distance = Inf
        for a in eachrow(A)
            d = distance_metric(a,b)
            distance = min(distance, d)
        end
        ba = max(ba, distance)
    end
    return max(ab, ba)
end


function fps(X, n_select; distance_metric = angular_distance)
    if n_select > length(X)
        println("The number of configurations to select is greater than the number of configurations, returning all ", length(X), " configurations")
        return 1:length(X)
    end

    if n_select == 0
        println("No points have been selected, as required")
        return Int64[]
    end

    keys = 1:length(X)
    selected = randperm(length(X))[1:1]
    D = vcat((X[k][3] for k in selected)...)
    remaining = setdiff(keys, selected)
    distance_vector = [minimum(
                        Hausdorff(X[s][3], X[k][3]; distance_metric=distance_metric)
                        for s in selected
                    ) for k in remaining]

    while length(selected) < n_select
        maximum_value, pos = maximum(distance_vector), argmax(distance_vector)
        max_index = remaining[pos]
        println("Maximum distance: ", maximum_value, " selected point ", max_index)
        push!(selected, max_index)
        remaining = deleteat!(remaining, pos)
        D = vcat(D, X[max_index][3])
        distance_vector = [minimum(
                        Hausdorff(X[s][3], X[k][3]; distance_metric=distance_metric)
                        for s in selected
                    ) for k in remaining]
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
    vect = [1.0 for _ in 1:nS]
    σ² = (e = (p = 1.0, s = 1. * vect), f = (p = 1.0, s = 1. * vect), v = (p = 1.0, s = 1.0 * vect)) 
    η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
    ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
    K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    kept_lines = filter_cov(K_no_hyper)
    return kept_lines
end

function extract_full_rank_submatrix(Y, train, K::AbstractMatrix, rtol=1e-8; verbose=true)
    n = size(K)[1]
    
    F = qr(K, ColumnNorm())        # RRQR: pivots ordered by information content
    r_diag = abs.(diag(F.R))
    tol = rtol * r_diag[1]
    r = count(r_diag .> tol)

    kept = sort(F.p[1:r])             # indices to keep, as ROW = COLUMN indices

    if verbose
        println("QR-pivoted rank = ", r, " / ", n, "  (dropped ", n - r, " indices)")
    end

    ne, nf, nv = BlockSizes(Y, train)
    sort_lines = sort(kept)

    true_ne = length(sort_lines[sort_lines .<= ne])
    println("We have kept ", true_ne, " values of energy over the ", ne, " available")

    return kept
end

function create_filter_cov(X, Y, train, nS, ζ, rtol; only_K=false)
    # careful, may be an issue with multitask as sigma is ill calibrated. Fix the selection in this case.
    vect = [1.0 * k for k in 1:nS]
    σ² = (e = (p = 1.0, s = vect), f = (p = 1.0, s = vect), v = (p = 1.0, s = vect)) 
    η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
    ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
    println(train, σ², ϱ, η)
    K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    if isempty(K_no_hyper)
        return []
    end
    kept_lines = extract_full_rank_submatrix(Y, train, K_no_hyper, rtol)
    if only_K
        return K_no_hyper, kept_lines
    end
    return kept_lines
end

function create_filter_cov_decoupled(X, Y, train, nS, ζ; only_K=false)
    vect_empty = [Float64[] for _ in 1:nS]
    train_e = (e = (p = train.e.p, s = train.e.s), f = (p = [], s = vect_empty), v = (p = [], s = vect_empty) )
    train_f = (e = (p = [], s = vect_empty), f = (p = train.f.p, s = train.f.s), v = (p = [], s = vect_empty) )
    kept_lines_e = create_filter_cov(X, Y, train_e, nS, ζ, 1e-8)
    kept_lines_f = create_filter_cov(X, Y, train_f, nS, ζ, 1e-8)
    ne, nf, nv = BlockSizes(Y, train)
    kept_lines_f = kept_lines_f .+ ne
    kept_lines_tot = union(kept_lines_e, kept_lines_f)
    println("We have kept ", length(kept_lines_tot), " data over the ", ne+nf+nv, " available")
    return kept_lines_tot
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

    # # ---- Row 1: parity plot -------------------------------------------------
    # ax1 = Axis(fig[1, col], xlabel = "True ($unit)", ylabel = "Predicted ($unit)",
    #     title = "Predicted vs. true ($(pd.label))",
    #     subtitle = @sprintf("RMSE = %.4g   ", pd.rmse),
    #     aspect = DataAspect())

    # errorbars!(ax1, pd.ytrue, pd.ypred, pd.sigma, pd.sigma, color = c_err, linewidth = 1, whiskerwidth = 0)
    # scatter!(ax1, pd.ytrue[idx1], pd.ypred[idx1], color = (c_good, 0.65), markersize = 9,
    #          strokewidth = 0.5, strokecolor = (:white, 0.6), label = "|z| ≤ 1")
    # scatter!(ax1, pd.ytrue[idx2], pd.ypred[idx2], color = (c_mid, 0.75), markersize = 9,
    #          strokewidth = 0.5, strokecolor = (:white, 0.6), label = "1 < |z| ≤ 2")
    # scatter!(ax1, pd.ytrue[idx3], pd.ypred[idx3], color = (c_bad, 0.85), markersize = 9,
    #          strokewidth = 0.5, strokecolor = (:white, 0.6), label = "|z| > 2")
    # lines!(ax1, pd.xr, pd.xr, color = c_accent, linestyle = :dash, linewidth = 2)
    # xlims!(ax1, pd.xr...); ylims!(ax1, pd.xr...)
    # axislegend(ax1, position = :rb, labelsize = 15)

    # # ---- Row 2: residuals vs. true ------------------------------------------
    # ax3 = Axis(fig[2, col], xlabel = "True ($unit)", ylabel = "Prediction error ($unit)",
    #     title = "Residuals ($(pd.label))")

    # scatter!(ax3, pd.ytrue[idx1], pd.resid[idx1], color = (c_good, 0.65), markersize = 8)
    # scatter!(ax3, pd.ytrue[idx2], pd.resid[idx2], color = (c_mid, 0.75), markersize = 8)
    # scatter!(ax3, pd.ytrue[idx3], pd.resid[idx3], color = (c_bad, 0.85), markersize = 8)
    # hlines!(ax3, [0.0], color = :black, linestyle = :dash, linewidth = 2)
    # xlims!(ax3, pd.xr...)

    # ---- Row 3: calibration / reliability, with Wilson sampling-noise band --
    ax4 = Axis(fig[1, col], xlabel = "Nominal confidence level", ylabel = "Empirical coverage",
        title = "Uncertainty calibration ($(pd.label))", aspect = DataAspect())

    levels = collect(union(0.05:0.05:0.95,0.99))
    observed = [Statistics.mean(abs.(pd.z) .<= norminvcdf(0.5 + p/2)) for p in levels]
    calib_err = Statistics.mean(abs.(observed .- levels))
    ax4.subtitle = @sprintf("Mean calibration error = %.3f   (N = %d)", calib_err, n)

    lines!(ax4, [0, 1], [0, 1], color = c_accent, linestyle = :dash, linewidth = 2, label = "Perfect calibration")
    lines!(ax4, levels, observed, color = c_bad, linewidth = 2.5)
    scatter!(ax4, levels, observed, color = c_bad, markersize = 8, label = "Model")
    xlims!(ax4, 0, 1); ylims!(ax4, 0, 1)
    axislegend(ax4, position = :lt, labelsize = 15)

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
function plot_predictions_pro(e_pred, e_true, std, f_pred = nothing, f_true = nothing, std_f = nothing;
                               model_name::String = "", save_path::Union{Nothing,String} = nothing)

    energy = PropertyDiag("energy", e_true, e_pred, std)
    if f_pred === nothing || f_true === nothing || std_f === nothing
        forces = nothing
    else
        forces = PropertyDiag("forces", f_true, f_pred, std_f)
    end

    # -------------------------------------------------------------------------
    # Palette & theme — clean "editorial" look
    # -------------------------------------------------------------------------
    c_bg     = :white
    c_grid   = "#E7E9EE"
    c_accent = "#264653"   # slate, used for diagonal / reference lines
    c_good   = "#2A9D8F"   # teal    -> |z| <= 1
    c_mid    = "#E9C46A"   # amber   -> 1 < |z| <= 2
    c_bad    = "#E76F51"   # coral   -> |z| > 2
    c_err    = (:gray40, 0.25)
    palette  = (c_bg = c_bg, c_grid = c_grid, c_accent = c_accent,
                c_good = c_good, c_mid = c_mid, c_bad = c_bad, c_err = c_err)

    set_theme!(Theme(
        fontsize = 18,
        font = "TeX Gyre Heros Makie",
        Axis = (
            # backgroundcolor = c_bg,
            xgridcolor = c_grid, ygridcolor = c_grid,
            xgridwidth = 1, ygridwidth = 1,
            topspinevisible = false, rightspinevisible = false,
            titlefont = "TeX Gyre Heros Bold Makie",
            titlesize = 19,
            subtitlefont = "TeX Gyre Heros Makie",
            subtitlesize = 17,
            subtitlecolor = (:black, 0.6),
        ),
        Legend = (framevisible = true, backgroundcolor = (:white, 0.8), padding = (8,8,8,8)),
    ))

    if isnothing(forces)
        fig = Figure(size = (750, 1500))
    else
        fig = Figure(size = (1100, 700))
    end



    add_property_panels!(fig, 1, energy, palette, "Ha/atom")

    if !isnothing(forces)
        add_property_panels!(fig, 2, forces, palette, "Ha/Bohr")
        Label(fig[0, 1:2],
            "Model performance — energies only",
            fontsize = 26,
            font = "TeX Gyre Heros Bold Makie")
    else
        Label(fig[0, 1],
            "Uncertainty calibration: dataset with forces",
            fontsize = 26, font = "TeX Gyre Heros Bold Makie")
    end

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 18)

    if !isnothing(save_path)
        save(save_path, fig, px_per_unit = 3)
    end

    return fig
end


function Schur_complement(K11inv, K12, K22)
    return K22 - K12' * K11inv * K12
end

function Schur_inversion(K11, K12, K22)
    L11 = cholesky(K11)
    Kinv = inv(L11)
    S = Schur_complement(Kinv, K12, K22)
    Ls = cholesky(Symmetric(S))
    Sinv = inv(Ls)
    KinvK12 = Kinv * K12
    TL = Kinv + KinvK12 * Sinv * KinvK12'
    TR = -KinvK12 * Sinv


    n = size(K11, 1)
    m = size(K22, 1)

    Kglob_inv = Matrix{Float64}(undef, n + m, n + m)
    Kglob_inv[1:n,     1:n]     .= TL
    Kglob_inv[1:n,     n+1:end] .= TR
    Kglob_inv[n+1:end, 1:n]     .= TR'
    Kglob_inv[n+1:end, n+1:end] .= Sinv
 
    return Kglob_inv
end
