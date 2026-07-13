using LinearAlgebra

struct HyperParams{T}
    σ²e::T
    σ²::Vector{T}
    ϱ::Vector{T}
    ηe::T
    ηf::T
    ηv::T
end

struct Size
    Se::Int64
    Sf::Int64
    Sv::Int64
end


struct CovMatrices
    Ce::Any
    Cf::Any
    Cv::Any
    Cfe::Any
    Cve::Any
    Cvf::Any
end

function BlockSizes(Y, trainP)
    ne = length(trainP.e.p) + sum(length(v) for v in trainP.e.s; init=0)
    nf = sum(length(Y[i].f) for i in trainP.f.p; init=0) + sum(length(Y[i].f) for v in trainP.f.s for i in v; init=0)
    nv = sum(length(Y[i].v) for i in trainP.v.p; init=0) + sum(length(Y[i].v) for v in trainP.v.s for i in v; init=0)
    return Size(ne, nf, nv)
end

function full_matrix(h::HyperParams, C::CovMatrices)
    nS = length(h.ϱ)
    vect_one = ones(nS)

    σ² = ( e=( p=h.σ²e, s=h.σ² ), f=( p=h.σ²e, s=h.σ² ), v=( p=h.σ²e, s=h.σ² ))
    ϱ  = ( e=h.ϱ, f=h.ϱ, v=h.ϱ ) #Be careful, rho has to be the same for e,f and v by linearity
    η  = ( e=( p=h.ηe, s=h.ηe * vect_one ), f=( p=h.ηf, s=h.ηf * vect_one ), v=( p=h.ηv, s=h.ηv * vect_one ))

    K = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true)

    Cho = cholesky(K)

    return Cho
end

function dnll(Cho, dK, y)
    α = Cho \ y
    return 0.5 * (tr(Cho \ dK) - dot(α, dK * α))
end

function dK_sigma_p(h::HyperParams, C)
    nS = length(h.ϱ)
    if nS == 0
        return C.p
    end
    p  = C.p
    ps = hcat([  C.ps[i]*h.ϱ[i]    for i in 1:nS ]...)

    if hasproperty(C, :sp)
        sp = vcat([  C.sp[i]*h.ϱ[i]    for i in 1:nS ]...)
        s  = vcat([   hcat([  C.s[i][j]*(h.ϱ[i]*h.ϱ[j])   for j in 1:nS]...)     for i in 1:nS]...)
        return [p ps ; sp s] 

    else
        s  = block_cat([   hcat([  C.s[i][j-i+1]*(h.ϱ[i]*h.ϱ[j])   for j in i:nS]...)     for i in 1:nS])
        return Symmetric( [p ps ; ps' s] )
    end
end

function dK_sigma_s(h::HyperParams, C, number_of_the_task)
    nS = length(h.ϱ)
    p  = 0 * C.p
    if nS == 0
        return p
    end
    ps = hcat([  C.ps[i]* 0    for i in 1:nS ]...)
    if hasproperty(C, :sp)
        sp = vcat([  C.sp[i]* 0    for i in 1:nS ]...)
        s = vcat([hcat([C.s[i][j] * (i == j == number_of_the_task) for j in 1:nS]...) for i in 1:nS]...)
        return [p ps ; sp s] 

    else
        s = block_cat([hcat([C.s[i][j-i+1] * (i == j == number_of_the_task) for j in i:nS]...) for i in 1:nS])
        return Symmetric( [p ps ; ps' s] )
    end
end

function dK_rho_s(h::HyperParams, C, number_of_the_task)
    nS = length(h.ϱ)

    p  = 0 * C.p    
    ps = hcat([  C.ps[i]*h.σ²e * (i == number_of_the_task)    for i in 1:nS ]...)
    if hasproperty(C, :sp)
        sp = vcat([  C.sp[i]*h.σ²e * (i == number_of_the_task)    for i in 1:nS ]...)
        s  = vcat([   hcat([  C.s[i][j]*( (i==number_of_the_task)*h.ϱ[j]*h.σ²e + (j == number_of_the_task)*h.ϱ[i]*h.σ²e)   for j in 1:nS]...)     for i in 1:nS]...)
        return [p ps ; sp s] 
    else
        s  = block_cat([   hcat([  C.s[i][j-i+1]*( (i==number_of_the_task)*h.ϱ[j]*h.σ²e + (j == number_of_the_task)*h.ϱ[i]*h.σ²e)   for j in i:nS]...)     for i in 1:nS])
        return Symmetric( [p ps ; ps' s] )
    end
end 

function nll(lh, C::CovMatrices, y, nS)
    σ²e = 10^lh[1]
    if nS == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+nS]
        ϱ = 10 .^lh[nS+2:2*nS+1]
    end
    ηe = 10^lh[2*nS+2]
    ηf = 10^lh[2*nS+3]
    ηv = 10^lh[2*nS+4]

    h = HyperParams(σ²e, σ², ϱ, ηe, ηf, ηv)  

    try
        Cho = full_matrix(h, C)
        return 0.5 * dot(y, Cho \ y) + sum(log.(diag(Cho.L)))
    catch
        println("inf detected, opinion rejected")
        println("Current hyperparameters: ", h)
        return Inf
    end
end


function dnll_compute(h::HyperParams, C, Cho, y; d_func)
    L = Any[]

    for name in fieldnames(typeof(C))
        val = getfield(C, name)
        push!(L, d_func(h, val))
    end
    dK = CovMatrices(L...)

    dK_constructed = [ dK.Ce dK.Cfe' dK.Cve' ; dK.Cfe dK.Cf dK.Cvf' ; dK.Cve dK.Cvf dK.Cv ]

    dnll_eps = dnll(Cho, dK_constructed, y)
    return dnll_eps
end



function g!(G, lh, C::CovMatrices, siz::Size, y, nS)

    σ²e = 10^lh[1]
    if nS == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+nS]
        ϱ = 10 .^lh[nS+2:2*nS+1]
    end
    ηe   = 10^lh[2*nS+2]               # scalaire, pas .^
    ηf   = 10^lh[2*nS+3]               # scalaire, pas .^
    ηv   = 10^lh[2*nS+4]               # scalaire, pas .^

    h   = HyperParams(σ²e, σ², ϱ, ηe, ηf, ηv)  # ← ordre correct (σ² avant ϱ !)


    Cho = try
        full_matrix(h, C)
    catch
        println("WARNING: inf detected in g!, gradient set to zero")
        fill!(G, 0.0)
        return
    end

    G[1] = dnll_compute(h, C, Cho, y; d_func = dK_sigma_p) * h.σ²e * log(10)

    for i in 1:nS
        G[1+i]    = dnll_compute(h, C, Cho, y;
                        d_func = (h, C) -> dK_sigma_s(h, C, i)
                    ) * h.σ²[i] * log(10)
    end
    for i in 1:nS
        G[1+nS+i] = dnll_compute(h, C, Cho, y;
                        d_func = (h, C) -> dK_rho_s(h, C, i)
                    ) * h.ϱ[i] * log(10)
    end
    dKe = Diagonal(vcat(ones(siz.Se), zeros(siz.Sf), zeros(siz.Sv)))
    dKf = Diagonal(vcat(zeros(siz.Se), ones(siz.Sf), zeros(siz.Sv)))
    dKv = Diagonal(vcat(zeros(siz.Se), zeros(siz.Sf), ones(siz.Sv)))

    G[2*nS+2] = dnll(Cho, dKe, y) * h.ηe * log(10)
    G[2*nS+3] = dnll(Cho, dKf, y) * h.ηf * log(10)
    G[2*nS+4] = dnll(Cho, dKv, y) * h.ηv * log(10)

end

function optim(C::CovMatrices, y, siz, nS)
    w = 2*nS + 4
    f(lh) = nll(vcat(lh), C, y, nS)
    G!(G, lh) = g!(G, vcat(lh), C, siz, y, nS)

    lower = fill(-6.0, w)
    upper = fill(6.0, w)
    p0 = vcat([0], fill(-4., nS), fill(0., nS), [-4,-4,-4])

    result = optimize(
        f, G!,
        lower, upper, p0,
        Fminbox(LBFGS(linesearch=Optim.LineSearches.HagerZhang())),
        Optim.Options(iterations=50, show_trace=true)
    )
    return result
end

function construct_hyperparam(res, number_of_tasks)
    vect_one = ones(number_of_tasks)

    σ²e = 10^res[1]

    if number_of_tasks == 0
        σ² = []
        ϱ = []
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

function Global_optimizer(X, Y, train, number_of_tasks, Ce, Cf, Cv, Cfe, Cve, Cvf, normalisation = false) 

    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)

    siz = BlockSizes(Y, train)

    y, m, s = select_observations(X, Y, train, normalisation)
    # print("OBSERVATIONS", y, "OK END")


    result = optim(C, y, siz, number_of_tasks)
    res = Optim.minimizer(result)
    nll_val = Optim.minimum(result)
    println("The optimisation obtained the nll value ", nll_val, " thanks to the parameters ", res)
    σ², ϱ, η = construct_hyperparam(vcat(res), number_of_tasks)
    
    return σ², ϱ, η

end

using ForwardDiff

function check_gradient_ad(C, y, siz, nS, lh)

    # analytique
    G = zeros(eltype(lh), length(lh))
    @time g!(G, lh, C, siz, y, nS)

    # autodiff
    f(x) = nll(x, C, y, nS)
    @time Gad = ForwardDiff.gradient(f, lh)

    println("Analytique : ", G)
    println("AD         : ", Gad)
    println("Erreur abs : ", abs.(G .- Gad))
    println("Erreur rel : ", abs.(G .- Gad) ./ max.(1e-12, abs.(Gad)))
end

function plot_nll(X, Y, train, nS, lh0, studied_hyper, low, up,
                  Ce, Cf, Cv, Cfe, Cve, Cvf, normalisation=false)

    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    x = range(low, up, length=200)
    y, _, _ = select_observations(X, Y, train, normalisation)

    lh0_values = Vector{Vector{Float64}}()
    for val in x
        params = copy(lh0)
        params[studied_hyper] = val
        push!(lh0_values, params)
    end

    nll_values = [nll(p, C, y, nS) for p in lh0_values]

    # minimum
    imin = argmin(nll_values)
    x_min = x[imin]
    nll_min = nll_values[imin]

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "log₁₀(σ²e)",
        ylabel = "NLL"
    )

    lines!(ax, x, nll_values)

    # point du minimum
    scatter!(ax, [x_min], [nll_min])

    # ligne verticale
    vlines!(ax, [x_min])

    # annotation
    text!(
        ax,
        x_min,
        nll_min;
        text = "min = $(round(x_min, digits=3))",
        align = (:left, :bottom)
    )

    display(fig)

    return x_min, nll_min
end

function plot_nll_2d(X, Y, train, nS, lh0, hyper_x, hyper_y,
                     low_x, up_x, low_y, up_y,
                     Ce, Cf, Cv, Cfe, Cve, Cvf;
                     n=80, normalisation=false)

    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    y, _, _ = select_observations(X, Y, train, normalisation)

    xs = range(low_x, up_x, length=n)
    ys = range(low_y, up_y, length=n)

    # Matrice NLL
    Z = [begin
            p = copy(lh0)
            p[hyper_x] = xi
            p[hyper_y] = yi
            nll(p, C, y, nS)
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
