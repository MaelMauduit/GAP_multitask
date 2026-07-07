using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask copy.jl")
include( path*"/linear_reg.jl")
include( path*"/Opti_all_in_one.jl")
include( path*"/tools.jl")

Xtrain26, Ytrain26, ltrain26 = read_data("/train26.xyz")
Xtrain10, Ytrain10, ltrain10 = read_data("/train10.xyz")
Xbcc26, Ybcc26, lb26 = read_data("/bcc26.xyz") 
Xbcc10, Ybcc10, lb10 = read_data("/bcc10.xyz")
Xtest10, Ytest10, lt10 = read_data("/bcc_dft_10.xyz")
Xtest, Ytest, lt = read_data("/bcc_dft_26.xyz")

# X = vcat(Xtrain26, Xtrain10, Xbcc26, Xbcc10, Xtest10, Xtest)
# X = X_normalise(X, train)
# Y = vcat(Ytrain26, Ytrain10, Ybcc26, Ybcc10, Ytest10, Ytest)

X = vcat(Xtrain26, Xbcc26, Xtest)
Y = vcat(Ytrain26, Ybcc26, Ytest)

# l26 = ltrain26
# l10 = l26+ltrain10
# lbcc26 = l10 + lb26
# lbcc10 = lbcc26 + lb10
# ltest10 = lbcc10+ lt10
# ltest = ltest10 + lt

l26 = ltrain26
lbcc26 = l26 + lb26
ltest = lbcc26 + lt

indx = (e = (p = union(1:lbcc26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx)

ζ=2
nS = 0
normalisation = true

# a = 1:lbcc26
# b = setdiff(1:l26, a)
# t = ltest10+1:4:ltest
a = fps(X[1:l26], 20)
b = fps(X[l26+1:end], 10)
# a = l10 .+ f
print(size(a[a .> l26]))
(lbcc26-l26)/l26

train = (e = (p = union(a,l26 .+b), s = []), f = (p = [], s = []), v = (p = [], s = []) )
trainfortry = (e = (p = union(a), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
test = (e = (p = lbcc26+1:ltest, s = []), f = (p = [], s = []), v = (p = [], s = []) )

# abc = (e = (p = union([]), s = []), f = (p = union(2), s = []), v = (p = union([]), s = []) )
# size(select_observations(X,Y,abc,false)[1])

Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train, ζ)
σ², ϱ, η = Global_optimizer(X, Y, train, nS, Ce, Cf, Cv, Cfe, Cve, Cvf)
# σ², ϱ, η = construct_hyperparam([σ².e.p, -7.84875565319201, -5.95518056787257, -0.17417136790277665], nS)

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

# Ybcc26[1].f
# # println(select_observations(X,Y,(e = (p = [], s = []), f = (p = [], s = []), v = (p = 1:l26, s = []) ),false))

# plot_nll(X, Y, train, nS, [4.5,0.,0.,0.], 1,
#             -8, 8,
#             Ce, Cf, Cv, Cfe, Cve, Cvf,
#             normalisation)
# plot_nll_2d(X, Y, train, nS, [0.,0.], 1, 2,
#             -8, 8, -8, 0,
#             Ce, Cf, Cv, Cfe, Cve, Cvf;
#             n=20, normalisation=normalisation)


# abc = (e = (p = union(a[1:2]), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
# def = (e = (p = union(a[1:2]), s = []), f = (p = union(a[1:2]), s = []), v = (p = union(a[1:2]), s = []) )
# Kabc = construct_covariance(X, abc, σ², ϱ, η, true, 4)
# Kdef = construct_covariance(X, def, σ², ϱ, η, true, 4)
# cond(Kdef)



# # energie seule
# train_e_only = (e = (p=union(a,l26 .+b), s=[]), f = (p=l26 .+b, s=[]), v = (p=l26 .+b, s=[]))
# Ce_e, Cf_e, Cv_e, Cfe_e, Cve_e, Cvf_e = construct_matrices(X, train_e_only, ζ)
# # σ², ϱ, η = Global_optimizer(X, Y, train_e_only, nS, Ce_e, Cf_e, Cv_e, Cfe_e, Cve_e, Cvf_e)
# # ϱ = (e = (p = 2.212693712828641e-8, s = Float64[]), f = (p = 0.00024965978761395713, s = Float64[]), v = (p = 0.0014315325549888376, s = Float64[])))
# m_e, _ = multitask(X, Y, train_e_only, test, σ², ϱ, η; ζ=ζ, normalisation=normalisation)
# println("R² énergie only = ", r2(m_e, truth))

# # tout
# train_all = (e = a[2], f = (p=a[2], s=[]), v = (p=a[2], s=[]))
# Ce_a, Cf_a, Cv_a, Cfe_a, Cve_a, Cvf_a = construct_matrices(X, train_all, ζ)
# σ²_a, ϱ_a, η_a = Global_optimizer(X, Y, train_all, nS, Ce_a, Cf_a, Cv_a, Cfe_a, Cve_a, Cvf_a)
# testforce = (e = (p = [], s = []), f = (p = lbcc26+1:ltest, s = []), v = (p = [], s = []) )
# m_all, _ = multitask(X, Y, train_all, testforce, σ²_a, ϱ_a, η_a; ζ=ζ, normalisation=normalisation) # WARNING takes above opti res
# println("R² tout         = ", r2(m_all, truth))

# # diagnostics
# K_all = construct_covariance(X, train_all, σ²_a, ϱ_a, η_a, true, ζ)
# ne, nf, nv = BlockSizes(Y, train_all)
# println("κ(K_ee) = ", cond(K_all[1:ne, 1:ne]))
# println("κ(K_ff) = ", cond(K_all[ne+1:ne+nf, ne+1:ne+nf]))
# println("κ(K_vv) = ", cond(K_all[ne+nf+1:end, ne+nf+1:end]))
# println("diag_mean ee = ", Statistics.mean(diag(K_all[1:ne, 1:ne])))
# println("diag_mean ff = ", Statistics.mean(diag(K_all[ne+1:ne+nf, ne+1:ne+nf])))
# println("diag_mean vv = ", Statistics.mean(diag(K_all[ne+nf+1:end, ne+nf+1:end])))


K = construct_covariance(X, train, σ², ϱ, η, true, ζ)
data, mean_for_1_atom, std_for_1_atom = select_observations(X, Y, train, normalisation)


if cond(K) > 1e10
    println("Warning: covariance matrix is ill-conditioned, results may be inaccurate as ", "cond(K) = ", cond(K))

else
    println("Covariance matrix is well-conditioned, proceeding with predictions")
end
Ktt = construct_covariance(X, test,  σ², ϱ, η, false, ζ)
Kt  = construct_covariance(X, train, test, σ², ϱ, σ², ϱ, ζ)

println("Filtering covariance matrix")
kept_lines = filter_cov(X, train)
K = K[kept_lines, kept_lines]
cond(K + Statistics.mean(diag(K))*1e-8*I)  # regularisation
Kt = Kt[kept_lines, :]
data = data[kept_lines]
println("new cond(K) = ", cond(K))
