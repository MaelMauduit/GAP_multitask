include( path*"/Loss_functions.jl")

using LinearAlgebra
using Optim

struct HyperParams{T}
    σ²e::T
    σ²::Vector{T}
    ϱ::Vector{T}
    ηe::T
    ηf::T
    ηv::T
end


struct CovMatrices
    Ce::Any
    Cf::Any
    Cv::Any
    Cfe::Any
    Cve::Any
    Cvf::Any
end

struct DataInfo
    Y::Any
    y::Any
    train::Any
    kept::Any
    nS::Int
    C::CovMatrices
    X::Any
end

function BlockSizes(Y, train)
    ne = length(train.e.p) + sum(length(v) for v in train.e.s; init=0)
    nf = sum(length(Y[i].f) for i in train.f.p; init=0) + sum(length(Y[i].f) for s in train.f.s for i in s; init=0)
    nv = sum(length(Y[i].v) for i in train.v.p; init=0) + sum(length(Y[i].v) for s in train.v.s for i in s; init=0)
    return ne, nf, nv
end

function safe_mean_diag(K_0, range)
    if isempty(range)
        return 1.0   
    end
    return Statistics.mean(diag(K_0[range, range]))
end

function get_jitter(h::HyperParams, Y, train, kept, K)
    """
    Must use K[kept_lines,kept_lines] as K. The kept_lines argument is used of 
    selecting lines, not acting on K.
    """
    ne, nf, nv = BlockSizes(Y, train)
    sort_lines = sort(kept)

    true_ne = length(sort_lines[sort_lines .<= ne])
    true_nf = length(sort_lines[(sort_lines .> ne) .& (sort_lines .<= ne + nf)])
    true_nv = length(sort_lines[sort_lines .> ne + nf])

    K_filtered = K[kept, kept]
    Me = safe_mean_diag(K_filtered, 1:true_ne)
    Mf = safe_mean_diag(K_filtered, true_ne+1:true_ne+true_nf)
    Mv = safe_mean_diag(K_filtered, true_ne+true_nf+1:true_ne+true_nf+true_nv)

    ηe = max(1e-7 * Me, 1e-4 * h.ηe)
    ηf = max(1e-7 * Mf, 1e-4 * h.ηf)
    ηv = max(1e-7 * Mv, 1e-4 * h.ηv)
    jitter = ((true_ne,ηe) , (true_nf,ηf) , (true_nv,ηv))
    return jitter
end

function full_matrix(h::HyperParams, Info::DataInfo)
    vect_one = ones(Info.nS)
    σ² = ( e=( p=h.σ²e, s=h.σ² ), f=( p=h.σ²e, s=h.σ² ), v=( p=h.σ²e, s=h.σ² ))
    ϱ  = ( e=h.ϱ, f=h.ϱ, v=h.ϱ ) #Be careful, rho has to be the same for e,f and v by linearity
    η  = ( e=( p=h.ηe, s=h.ηe * vect_one ), f=( p=h.ηf, s=h.ηf * vect_one ), v=( p=h.ηv, s=h.ηv * vect_one ))

    C = Info.C
    K = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true, Info.X, Info.train)

    jitter  = get_jitter(h, Info.Y, Info.train, Info.kept, K)
    ne, ηe = jitter[1]
    nf, ηf = jitter[2]
    nv, ηv = jitter[3]

    K = K[Info.kept, Info.kept]

    D = Diagonal(vcat(fill(ηe, ne), fill(ηf, nf), fill(ηv, nv)))

    Cho = cholesky(K + D)
    return Cho
end


function optim(Info, estimator)
    println("Starting the optimization with optimizer: LBFGS")
    nS = Info.nS
    w = 2*nS + 4
    methods = Dict(
        "nll" => nll,
        "map" => map,
        "pnll" => pnll,
    )   
    f(lh) = methods[estimator](vcat(lh), Info)

    lower = fill(-8.0, w)
    upper = fill(8.0, w)
    p0 = vcat([-0.], fill(-4., nS), fill(0., nS), [-5., -5., -5.])

    result = optimize(
        f,
        lower, upper, p0,
        Fminbox(
            LBFGS(
                linesearch = Optim.LineSearches.HagerZhang()
            )
        ),
        Optim.Options(
            iterations = 50,
            show_trace = true
        ),
        autodiff = :forward
    )
    return result
end


function optim2(Info, estimator)
    println("Starting the optimization with optimizer: IPNewton")
    nS = Info.nS
    w = 2*nS + 4
    methods = Dict(
        "nll"  => nll,
        "map"  => map,
        "pnll" => pnll,
    )
    f(lh) = methods[estimator](vcat(lh), Info)

    lower = fill(-8.0, w)
    upper = fill(8.0, w)
    p0 = vcat([-0.], fill(-4., nS), fill(0., nS), [-5., -5., -5.])

    df  = TwiceDifferentiable(f, p0; autodiff = :forward)
    dfc = TwiceDifferentiableConstraints(lower, upper)

    result = optimize(
        df, dfc, p0,
        IPNewton(),
        Optim.Options(
            iterations = 50,
            show_trace = true
        )
    )
    return result
end


function construct_hyperparam(res, number_of_tasks)
    vect_one = ones(number_of_tasks)

    σ²e = 10^res[1]

    if number_of_tasks == 0
        σ² = Float64[]
        ϱ = Float64[]
    else 
        σ² = 10 .^res[2:1+number_of_tasks]
        ϱ = 10 .^res[number_of_tasks+2:2*number_of_tasks+1]
    end
    ηe = 10^res[2*number_of_tasks+2]
    ηf = 10^res[2*number_of_tasks+3]
    ηv = 10^res[2*number_of_tasks+4]

    σ² = ( e=( p=σ²e, s=σ² ), f=( p=σ²e, s=σ² ), v=( p=σ²e, s=σ² ))
    ϱ = (e = ϱ, f = ϱ, v = ϱ)
    η  = ( e=( p=ηe, s=ηe * vect_one ), f=( p=ηf, s=ηf * vect_one ), v=( p=ηv, s=ηv * vect_one ))
    return σ², ϱ, η
end

function Global_optimizer(X, Y, train, kept_lines, number_of_tasks; estimator = "nll", normalisation = false, ζ = 4) 
    Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train, ζ)
    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    y, _, _ = select_observations(X, Y, train, normalisation)
    y = y[kept_lines] 

    Info = DataInfo(Y, y, train, kept_lines, number_of_tasks, C, X)

    result = optim2(Info, estimator)
    res = Optim.minimizer(result)
    nll_val = Optim.minimum(result)
    println("The optimisation obtained the nll value ", nll_val, " thanks to the parameters ", res, "with a final gradient of ")
    σ², ϱ, η = construct_hyperparam(res, number_of_tasks)

    return σ², ϱ, η
end



function plot_nll_2d(X, Y, train, kept_lines, number_of_tasks,
                     lh0, hyper_x, hyper_y,
                     low_x, up_x, low_y, up_y;
                     n=20, normalisation=true, ζ = 4)

    Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train, ζ)
    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    y, _, _ = select_observations(X, Y, train, normalisation)
    y = y[kept_lines] 

    Info = DataInfo(Y, y, train, kept_lines, number_of_tasks, C, X)

    xs = range(low_x, up_x, length=n)
    ys = range(low_y, up_y, length=n)

    # Matrice NLL
    Z = [begin
            p = copy(lh0)
            p[hyper_x] = xi
            p[hyper_y] = yi
            nll(p, Info)
         end
         for yi in ys, xi in xs]   # shape: (n, n) — lignes=y, colonnes=x

    # Minimum
    imin = argmin(Z)
    x_min = xs[imin[2]]
    y_min = ys[imin[1]]
    nll_min = Z[imin]

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "log₁₀(σ²e)",
        ylabel = "log₁₀(σ²f)",
        title  = "NLL — nS=0"
    )

    hm = heatmap!(ax, xs, ys, Z')   # transposée pour aligner axes
    Colorbar(fig[1, 2], hm, label = "NLL")

    # Croix au minimum
    scatter!(ax, [x_min], [y_min],
             marker = :cross, markersize = 15,
             color = :white, strokewidth = 1)

    text!(ax, x_min, y_min;
          text   = "min=($(round(x_min,digits=2)), $(round(y_min,digits=2)))",
          align  = (:left, :bottom),
          color  = :white,
          fontsize = 11)

    display(fig)
    return x_min, y_min, nll_min
end