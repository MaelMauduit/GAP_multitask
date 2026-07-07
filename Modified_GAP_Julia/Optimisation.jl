using FiniteDifferences
struct HyperParams
    σ²e::Float64
    ηe::Float64
    ηf::Float64
    ηv::Float64
end



HyperParams(p::AbstractVector) = HyperParams(p[1], p[2], p[3], p[4])
to_vector(h::HyperParams) = [h.σ²e, h.ηe, h.ηf, h.ηv]

struct CovMatrices
    Ce::Any
    Cf::Any
    Cv::Any
    Cfe::Any
    Cve::Any
    Cvf::Any
end

struct Sizes
    ne :: Int
    nf :: Int
    nv :: Int
end
Base.iterate(s::Sizes, state=1) = state > 3 ? nothing : (getfield(s, state), state+1)

function BlockSizes(Y, trainP)
    ne = length(trainP.e.p) + sum(length(v) for v in trainP.e.s; init=0)
    nf = sum(length(Y[i].f) for i in trainP.f.p; init=0) + sum(length(Y[i].f) for v in trainP.f.s for i in v; init=0)
    nv = sum(length(Y[i].v) for i in trainP.v.p; init=0) + sum(length(Y[i].v) for v in trainP.v.s for i in v; init=0)
    return Sizes(ne, nf, nv)
end

function full_matrix(h::HyperParams, C::CovMatrices, secondary_task_index, number_of_tasks, ϱs = nothing, σp = nothing)
    vect_one = ones(number_of_tasks)
    if secondary_task_index == 0
        σ² = ( e=( p=h.σ²e, s=0*vect_one ), f=( p=h.σ²e, s=0*vect_one ), v=( p=h.σ²e, s=0*vect_one ))
        ϱ  = ( e=vect_one, f=vect_one, v=vect_one )
        η  = ( e=( p=h.ηe, s=1e-5 * vect_one ), f=( p=h.ηf, s=1e-5 * vect_one ), v=( p=h.ηv, s=1e-5 * vect_one ))
    else
        s_σ² = 0 * vect_one; s_σ²[secondary_task_index] = h.σ²e
        s_ϱ  = vect_one ; s_ϱ[secondary_task_index]  = ϱs
        s_ηe = 1e-5 * vect_one; s_ηe[secondary_task_index] = h.ηe
        s_ηf = 1e-5 * vect_one; s_ηf[secondary_task_index] = h.ηf
        s_ηv = 1e-5 * vect_one; s_ηv[secondary_task_index] = h.ηv
        σ² = ( e=( p=σp, s=copy(s_σ²) ), f=( p=σp, s=copy(s_σ²) ), v=( p=σp, s=copy(s_σ²) ))
        ϱ  = ( e=copy(s_ϱ), f=copy(s_ϱ), v=copy(s_ϱ) )
        η  = ( e=( p=1e-5, s=s_ηe ), f=( p=1e-5, s=s_ηf ), v=( p=1e-5, s=s_ηv ))
    end

    K = hyperparameters(C.Ce, C.Cf, C.Cv, C.Cfe, C.Cve, C.Cvf, σ², ϱ, η, true)

    # println("K size: ", size(K))
    # println("K symmetry error: ", norm(K - K'))
    # println("K min diag: ", minimum(diag(K)))
    # println("K λmin: ", minimum(eigvals(Symmetric(K))))    # Debug
    # λmin = minimum(eigvals(Symmetric(K)))
    # @printf("σ²e=%.4e  ηe=%.4e  ηf=%.4e  ηv=%.4e  λmin=%.4e\n", h.σ²e, h.ηe, h.ηf, h.ηv, λmin)

    Cho = cholesky(K)
    return K, Cho
end

function nll(h::HyperParams, C::CovMatrices, y, secondary_task_index, number_of_tasks, ϱs = nothing, σp = nothing)
    _, Cho = full_matrix(h, C, secondary_task_index, number_of_tasks, ϱs, σp)
    return 0.5 * dot(y, Cho \ y) + sum(log.(diag(Cho.L)))
end

function dnll_sigma(h::HyperParams, siz::Sizes, K, Cho, y, secondary_task_index, ϱs, σp)
    ne, nf, nv = siz
    η_diag = Diagonal([fill(h.ηe, ne); fill(h.ηf, nf); fill(h.ηv, nv)])
    
    if secondary_task_index == 0
        dK_dσ² = (K - η_diag) / h.σ²e
    else
        Fact = h.σ²e + ϱs^2 * σp 
        dK_dσ² = (K - η_diag) / Fact      
    end
    α = Cho \ y
    return 0.5 * (tr(Cho \ dK_dσ²) - dot(α, dK_dσ² * α))
end

function dnll_eta(siz::Sizes, quantity, Cho, y)
    ne, nf, nv = siz
    if quantity == "e"
        dK = Diagonal([fill(1., ne); fill(0., nf); fill(0., nv)])
    elseif quantity == "f"
        dK = Diagonal([fill(0., ne); fill(1., nf); fill(0., nv)])
    elseif quantity == "v"
        dK = Diagonal([fill(0., ne); fill(0., nf); fill(1., nv)])
    end
    α = Cho \ y
    return 0.5 * (tr(Cho \ dK) - dot(α, dK * α))
end

function nll_log(logp, C::CovMatrices, y, secondary_task_index, number_of_tasks, ϱs = nothing, σp = nothing)
    h = HyperParams(10 .^ logp)
    return nll(h, C, y, secondary_task_index, number_of_tasks, ϱs, σp)
end

function g!(G, logp, C::CovMatrices, siz::Sizes, y, secondary_task_index, number_of_tasks, ϱs = nothing, σp = nothing)
    h = HyperParams(10 .^ logp)
    K, Cho = full_matrix(h, C, secondary_task_index, number_of_tasks, ϱs, σp)
    G[1] = dnll_sigma(h, siz, K, Cho, y, secondary_task_index, ϱs, σp) * h.σ²e * log(10)
    G[2] = dnll_eta(siz, "e", Cho, y) * h.ηe  * log(10)
    G[3] = dnll_eta(siz, "f", Cho, y) * h.ηf  * log(10)
    G[4] = dnll_eta(siz, "v", Cho, y) * h.ηv  * log(10)
end

function insert_at(vec, i)
    s = [[] for _ in 1:length(vec)]  
    s[i] = vec[i]
    return s
end
 
function plot_nll_sigma(C, y, secondary_task_index, number_of_tasks, ϱs, σp, ηe, ηf, ηv)
    x = range(-1, 1, length=50)
    nll_values = [nll_log([logσ², log10(ηe), log10(ηf), log10(ηv)], C, y, secondary_task_index, number_of_tasks, ϱs, σp) 
                  for logσ² in x]
    
    fig, ax, _ = lines(x, nll_values)
    ax.xlabel = "log₁₀(σ²e)"
    ax.ylabel = "NLL"
    display(fig)
end

function optim(X, Y, train_dataset, secondary_task_index, number_of_tasks, ζ = 4., ϱs = nothing, σp = nothing, plot = false)
    if secondary_task_index == 0
        Sub_dataset = (
            e = ( p=train_dataset.e.p, s=[ [], [] ] ),
            f = ( p=train_dataset.f.p, s=[ [], [] ] ),
            v = ( p=train_dataset.v.p, s=[ [], [] ] )
        )
    else
        @assert ϱs !== nothing "ϱs must have a value"
        @assert σp !== nothing "σp must have a value"

        Sub_dataset = (
            e = ( p=[], s=insert_at(train_dataset.e.s, secondary_task_index) ),
            f = ( p=[], s=insert_at(train_dataset.f.s, secondary_task_index) ),
            v = ( p=[], s=insert_at(train_dataset.v.s, secondary_task_index) )
        ) 
    end
    println(Sub_dataset)
    Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, Sub_dataset, ζ)
    C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    y, _, _ = select_observations(Y, Sub_dataset, false)
    siz = BlockSizes(Y, Sub_dataset)

    
    f(logp)       = nll_log(logp, C, y, secondary_task_index, number_of_tasks, ϱs, σp)
    g_S!(G, logp) = g!(G, logp, C, siz, y, secondary_task_index, number_of_tasks, ϱs, σp)

    lower = fill(-10.0, 4)
    upper = fill( 10.0, 4)
    p0 = [2., -3., -3., -3.]
    nll_history = []

    result = optimize(
        f,
        g_S!,
        lower,
        upper,
        p0,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations = 1000,
            g_tol      = 1e-5,
            store_trace = true,
            extended_trace = true,
            callback = state -> (push!(nll_history, state[end].value); false)        )
    )
    # trace = Optim.trace(result)
    # nll_history = [t.value for t in trace]    
    # # ln(trace[120], nll_history[120], trace[600], nll_history[600])
    # fig, ax, _ = lines(1:length(nll_history), nll_history[1:end])
    # ax.xlabel = "itérations"
    # ax.ylabel = "NLL"
    # display(fig)

    Res = 10 .^ Optim.minimizer(result)
    ηe, ηf, ηv = Res[2:4]

    if plot
        plot_nll_sigma(C, y, secondary_task_index, number_of_tasks, ϱs, σp, ηe, ηf, ηv)
    end
    return Res
end


