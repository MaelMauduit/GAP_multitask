using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask copy.jl")
include( path*"/Clean_optimisation.jl")
include( path*"/tools.jl")


Xtrain26, Ytrain26, ltrain26 = read_data("/train26.xyz")
Xbcc26, Ybcc26, lb26 = read_data("/bcc26.xyz") 
Xtest, Ytest, lt = read_data("/bcc_dft_26.xyz")

X = vcat(Xtrain26, Xbcc26, Xtest)
Y = vcat(Ytrain26, Ybcc26, Ytest)

train_features = fps(X[1:ltrain26], 6546)
bcc_features = ltrain26 .+fps(X[ltrain26+1:ltrain26+lb26], 54400)
test_features = ltrain26+lb26+1:ltrain26 + lb26 + lt 

ζ=2
nS = 0
normalisation = true


indx = (e = (p = union(1:ltrain26+lb26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx)

# train = (e = (p = union(bcc_features, test_features[1]), s = []), f = (p = union(bcc_features, test_features[1]), s = []), v = (p = union(bcc_features, test_features[1]), s = []))
train = (e = (p = union(bcc_features, test_features[1]), s = []), f = (p = [], s = []), v = (p = [], s = []))



test = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []) )
# m, S = multitask( X, Y, train, test, σ², ϱ, η; ζ = ζ, normalisation = normalisation, filter = true)

m, S = multitask( X, Y, train, test, 0; ζ = ζ, normalisation = normalisation)
Var = diag(S)

truth, mean, std = select_observations( X, Y, test, false)
ε     = r2( m, truth )
print("R² = ", ε, "\n")

Volume = [Ytest[k].volume[1] for k in 1:length(Ytest)]
yb, _, _ = select_observations(Xtest, Ytest, (e = (p = 1:lt, s = []), f = (p = [], s = []), v = (p = [], s = []) ), false)
plot_result(m, Var, truth, Volume, union([]), ε; 
    savedir = nothing)

