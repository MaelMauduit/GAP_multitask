using LinearAlgebra
using Random
using Distributions
using Tullio

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
Xbcc26, Ybcc26, lb26 = read_data("/bcc26.xyz") 
Xtest, Ytest, lt = read_data("/bcc_dft_26.xyz")

X = vcat(Xtrain26, Xbcc26, Xtest)
Y = vcat(Ytrain26, Ybcc26, Ytest)
indx = (e = (p = union(1:ltrain26+lb26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx)
norm(X[1][3][3,:], 2)
small_X = X[1:2]
# ζ=2
# nS = 0
# normalisation = true

key = (e = (p = [], s = []), f = (p = 1, s = []), v = (p = [], s = []))
η = (e = (p = 1e-8, s = Float64[]), f = (p = 1e-8, s = Float64[]), v = (p = 1e-8, s = Float64[]))
ϱ = (e = (p = 1.0, s = Float64[]), f = (p = 1.0, s = Float64[]), v = (p = 1.0, s = Float64[]))
σ² = (e = (p = 1.0, s = Float64[]), f = (p = 1.0, s = Float64[]), v = (p = 1.0, s = Float64[]))

construct_covariance( small_X, key, σ², ϱ, η, true, 2)

function grad(small_X)
    X = small_X[3]
    dX = small_X[1]
    val = 0
    for i in 1:size(X,1)
        for j in 1:size(X,1)

            dC = ζ*(ζ-1)*(last(X)' * last(X)).^(ζ-2)
            println(size(dC))
            println(size(dX[i,1,1,:]))
            plus = (dX[i,1,1,:] * dC[i,j] * dX[j,1,1,:]')
            val = val + plus
        end
    end
    return val

end
grad(small_X)