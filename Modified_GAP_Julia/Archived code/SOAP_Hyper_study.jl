# using Statistics
# using MultivariateStats
# using LinearAlgebra, Distributions
# using GLMakie

# path  = @__DIR__

# include( path*"/multitask copy.jl")
# include( path*"/linear_reg.jl")
# include( path*"/Opti_all_in_one.jl")
# include( path*"/tools.jl")

# Xtrain26, Ytrain26, ltrain26 = read_data("/train26.xyz")
# Xbcc26, Ybcc26, lb26 = read_data("/bcc26.xyz") 
# Xtest, Ytest, lt = read_data("/bcc_dft_26.xyz")

# X = vcat(Xtrain26, Xbcc26, Xtest)
# Y = vcat(Ytrain26, Ybcc26, Ytest)
# indx = (e = (p = union(1:ltrain26+lb26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
# X = X_normalise(X, indx)

# a = fps(X[1:ltrain26], 100)
# b = fps(X[ltrain26+1:ltrain26+lb26], 20)
# c = setdiff(ltrain26+1:(ltrain26 + lb26), b)
# ζ=2
# nS = 0
# normalisation = true

Ce_e, Cf_e, Cv_e, Cfe_e, Cve_e, Cvf_e = construct_matrices(X, train_e_only, ζ)
σ², ϱ, η = Global_optimizer(X, Y, train_e_only, nS, Ce_e, Cf_e, Cv_e, Cfe_e, Cve_e, Cvf_e)
train_e_only = (e = (p = union(a,ltrain26 .+b), s = []), f = (p = [], s = []), v = (p = [], s = []) )
train_f_only = (f = (p = union((a)[3]), s = []), e = (p = a[3], s = []), v = (p = [], s = []) )
train_v_only = (v = (p = union(a[2]), s = []), e = (p = [], s = []), f = (p = a[2], s = []) )
test = (e = (p = ltrain26+lb26+1:ltrain26 + lb26 + lt, s = []), f = (p = [], s = []), v = (p = [], s = []) )
train = (e = (p=union(a,ltrain26 .+b), s=[]), f = (p=ltrain26 .+b, s=[]), v = (p=ltrain26 .+b, s=[]))

η = (e = (p = 0, s = Float64[]), f = (p = 0, s = Float64[]), v = (p = 0, s = Float64[]))
K = construct_covariance(X, train, σ², ϱ, η, true, ζ)
ne, nf, nv = BlockSizes(Y, train)
Me = Statistics.mean(diag(K[1:ne, 1:ne]))
Mf = Statistics.mean(diag(K[ne+1:ne+nf, ne+1:ne+nf]))
Mv = Statistics.mean(diag(K[ne+nf+1:ne+nf+nv, ne+nf+1:ne+nf+nv]))
ηe = 1e-9 * Me
ηf = 1e-6* Mf
ηv = 1e-6* Mv 

η = (e = (p = ηe, s = Float64[]), f = (p = ηf, s = Float64[]), v = (p = ηv, s = Float64[]))
K = construct_covariance(X, train, σ², ϱ, η, true, ζ)

cond(K)
# m_e, _ = multitask(X, Y, train_e_only, test, σ², ϱ, η; ζ=ζ, normalisation=false)
# println("R² énergie only = ", r2(m_e, truth))

# m_e, _ = multitask(X, Y, train, test, σ², ϱ, η; ζ=ζ, normalisation=false)
# println("R² for all data = ", r2(m_e, truth))

# covariances construites sur Y_norm
K = construct_covariance(X, train_f_only, σ², ϱ, η, true, ζ)
λ, V = eigen(Symmetric(K))   # mieux que eigvals(K)
U, S, V = svd(Matrix(K))

rank = findfirst(<(1e-10), S ./ maximum(S))
println(rank, " over ", size(K)[1])
# tri décroissant
idx = sortperm(λ, rev=true)
λ = λ[idx]
V = V[:, idx]

# =========================
# 1. Eigenvalue spectrum
# =========================

fig = Figure(size = (900, 400))
rlambda = findfirst(<(1.), λ)

ax1 = Axis(fig[1, 1],
    title = "Eigenvalue spectrum",
    xlabel = "index",
    ylabel = "λ",
    yscale = log10
)

lines!(ax1, λ)
scatter!(ax1, λ)

# =========================
# 2. Cumulative variance
# =========================

cumulative = cumsum(λ) ./ sum(λ)
r99 = findfirst(>(0.99), cumulative)

println("Effective rank (99%): ", r99)

display(fig)
wait(fig.scene)

println("conditioning: ", cond(K), " with a smallest eigenvalue of ", minimum(λ), " noises hyper of ", η.e.p, " ", η.f.p, " ", η.v.p," and a largest eigenvalue of ", maximum(λ))
println("cumulative part of the first n values = ", cumulative[1:r99], " with 0.99 reached at ", r99, "th lambda")
println("Number of eigenvalues = ",
        length(λ))
println("Fraction below 1e-2 λmax = ",
        Statistics.mean(λ .< 1e-2*maximum(λ)))

println("Fraction below 1e-3 λmax = ",
        Statistics.mean(λ .< 1e-3*maximum(λ)))

println("Fraction below 1e-4 λmax = ",
        Statistics.mean(λ .< 1e-4*maximum(λ)))


A = Matrix(K)
A = A ./ sqrt.(sum(A.^2, dims=2))
n = size(A,1)

for i in 1:n
    for j in i+1:n
        sim = dot(A[i,:], A[j,:])
        if sim > 0.999
            println("doublon: ", i, " ", j, " sim=", sim)
        end
    end
end


train = (e = (p=union(ltrain26 .+b), s=[]), f = (p=ltrain26 .+b, s=[]), v = (p=ltrain26 .+b, s=[]))

cond(K)
kept_lines = filter_cov(X, train)
K_filtered = K[kept_lines, kept_lines]
print(K_filtered)
K_filtered
λ, V = eigen(Symmetric(K_filtered))   # mieux que eigvals(K)
U, S, V = svd(Matrix(K_filtered))
print(λ)
rank = findfirst(<(1e-10), S ./ maximum(S))
println(rank, " over ", size(K_filtered)[1])
println("Condition number: ", cond(K_filtered))

m, S = multitask( X, Y, train, test, σ², ϱ, η; ζ = ζ, normalisation = normalisation, filter = true)
# m, S = multitask( X, Y, trainfortry, test, σ², ϱ, η; ζ = ζ, normalisation = normalisation)
Var = diag(S)

# truth, mean, std = select_observations( Xbcc26, Ybcc26, (e = (p = 1:1, s = []), f = (p = 1:1, s = []), v = (p = [], s = []) ), false)
# print(truth[1:1])
# print(truth[2:end])
truth, mean, std = select_observations( X, Y, test, false)
ε     = r2( m, truth )
print("R² = ", ε, "\n")

Volume = [Ytest[k].volume[1] for k in 1:length(Ytest)]
yb, _, _ = select_observations(Xtest, Ytest, (e = (p = 1:lt, s = []), f = (p = [], s = []), v = (p = [], s = []) ), false)
plot_result(m, Var, truth, Volume, union([]), ε; 
    hyperparams = Dict("σ²e" => σ².e.p), 
    savedir = nothing)