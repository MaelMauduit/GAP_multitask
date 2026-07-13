desc = SOAPDescriptor(species=["Si"], r_cut=10.0, n_max=12, l_max=6, sigma=0.5)

function read_data(file)
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

function create_filter_cov(X,train,nS)
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

function plot_predictions(y_pred, y_true, number_of_atoms = nothing)
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

    xlims!(ax, lo - pad, hi + pad)
    ylims!(ax, lo - pad, hi + pad)
    axislegend(ax, position = :rb)

    fig
end

