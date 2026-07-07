using LinearAlgebra
using Optim

struct HyperParams{T}
    σ²e::T
    σ²::Vector{T}
    ϱ::Vector{T}
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
    return ne, nf, nv
end

function get_jitter(h::HyperParams, K_0, Y, train, kept_lines)
    nS = length(h.ϱ)
    vect_one = ones(nS)
    ne, nf, nv = BlockSizes(Y, train)
    sort_lines = sort(kept_lines)
    true_ne = length(sort_lines[sort_lines .<= ne])
    true_nf = length(sort_lines[(sort_lines .> ne) .& (sort_lines .<= ne + nf)])
    true_nv = length(sort_lines[sort_lines .> ne + nf])
    Me = Statistics.mean(diag(K_0[1:true_ne, 1:true_ne]))
    Mf = Statistics.mean(diag(K_0[true_ne+1:true_ne+true_nf, true_ne+1:true_ne+true_nf]))
    Mv = Statistics.mean(diag(K_0[true_ne+true_nf+1:true_ne+true_nf+true_nv, true_ne+true_nf+1:true_ne+true_nf+true_nv]))
    ηe = 1e-9 * Me
    ηf = 1e-6* Mf
    ηv = 1e-6* Mv 
    η  = ( e=( p=ηe, s=ηe * vect_one ), f=( p=ηf, s=ηf * vect_one ), v=( p=ηv, s=ηv * vect_one ))
    return η
end

function full_matrix(h::HyperParams, C::CovMatrices, Y, train, kept_lines)
    nS = length(h.ϱ)
    vect_one = ones(nS)
    σ² = ( e=( p=h.σ²e, s=h.σ² ), f=( p=h.σ²e, s=h.σ² ), v=( p=h.σ²e, s=h.σ² ))
    ϱ  = ( e=h.ϱ, f=h.ϱ, v=h.ϱ ) #Be careful, rho has to be the same for e,f and v by linearity
    ηe = 0
    η  = ( e=( p=ηe, s=ηe * vect_one ), f=( p=ηe, s=ηe * vect_one ), v=( p=ηe, s=ηe * vect_one ))
    K_0 = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true)
    K_0 = K_0[kept_lines, kept_lines]
    η  = get_jitter(h, K_0, Y, train, kept_lines)

    K = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true)
    K = K[kept_lines, kept_lines]
    # print("Me = ", Me, ", Mf = ", Mf, ", Mv = ", Mv)
    # print("cond(K) = ", cond(K), ", cond(K1) = ", cond(K1))
    Cho = cholesky(K)
    return Cho
end


function nll(lh, C::CovMatrices, y, number_of_tasks, Y, train, kept_lines)
    σ²e = 10^lh[1]
    if number_of_tasks == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+number_of_tasks]
        ϱ = 10 .^lh[number_of_tasks+2:2*number_of_tasks+1]
    end
    # ηe = 10^lh[2*nS+2]
    # ηf = 10^lh[2*nS+3]
    # ηv = 10^lh[2*nS+4]

    h = HyperParams(σ²e, σ², ϱ)  
    y = y[kept_lines]
    try
        Cho = full_matrix(h, C, Y, train, kept_lines)
        return 0.5 * dot(y, Cho \ y) + sum(log.(diag(Cho.L)))
    catch
        println("inf detected, opinion rejected")
        println("Current hyperparameters: ", h)
        return Inf
    end
end


function optim(C::CovMatrices, y, nS, train, kept_lines)
    w = 2*nS + 1
    f(lh) = nll(vcat(lh), C, y, nS, Y, train, kept_lines)

    lower = fill(-8.0, w)
    upper = fill(8.0, w)
    p0 = vcat([-1], fill(-4., nS), fill(0., nS))

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

function Global_optimizer(X, Y, train, kept_lines, number_of_tasks,  normalisation = false, ζ = 4) 
    Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train, ζ)
    vect_one = ones(number_of_tasks)
    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    ne, nf, nv = BlockSizes(Y, train)

    y, _, _ = select_observations(X, Y, train, normalisation)
    # print("OBSERVATIONS", y, "OK END")


    result = optim(C, y, number_of_tasks, train, kept_lines)
    res = Optim.minimizer(result)
    nll_val = Optim.minimum(result)
    println("The optimisation obtained the nll value ", nll_val, " thanks to the parameters ", res, "with a final gradient of ")
    
    ne, nf, nv = BlockSizes(Y, train)
    σ², ϱ, η = construct_hyperparam(vcat(res, [log10(1e-15), log10(1e-15), log10(1e-15)]), number_of_tasks)
    ηe = 0
    η  = ( e=( p=ηe, s=ηe * vect_one ), f=( p=ηe, s=ηe * vect_one ), v=( p=ηe, s=ηe * vect_one ))
    K_0 = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true)
    h = HyperParams(σ².e.p, σ².e.s, ϱ.e)
    η  = get_jitter(h, K_0, Y, train, kept_lines)
    println 
    return σ², ϱ, η

end