using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask copy.jl")
include( path*"/Clean_optimisation.jl")
include( path*"/tools.jl")


Xtrain26, Ytrain26, ltrain26 = read_data("/train26.xyz") ## Careful to the used dataset
Xbcc26, Ybcc26, lb26 = read_data("/bcc26.xyz") 
Xtest, Ytest, lt = read_data("/bcc_dft_26.xyz")

X = vcat(Xtrain26, Xbcc26, Xtest)
Y = vcat(Ytrain26, Ybcc26, Ytest)

indx = (e = (p = union(1:ltrain26+lb26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx)


train_features = fps(X[1:ltrain26], (1*ltrain26)//4)
# bcc_features = ltrain26 .+fps(X[ltrain26+1:ltrain26+lb26], 54400)
# test_features = ltrain26+lb26+1:ltrain26 + lb26 + lt 
test_features = setdiff(1:ltrain26, train_features)
ζ=2
nS = 0
normalisation = true



train = (e = (p = union(train_features), s = []), v = (p = [], s = []), f = (p = train_features, s = []))
# train = (e = (p = union(bcc_features, test_features[1]), s = []), f = (p = [], s = []), v = (p = [], s = []))

test_e = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []) )
test_f = (e = (p = [], s = []), f = (p = test_features, s = []), v = (p = [], s = []) )
test_e_and_f = (e = (p = test_features, s = []), f = (p = test_features, s = []), v = (p = [], s = []) )

# m, S = multitask( X, Y, train, test, σ², ϱ, η; ζ = ζ, normalisation = normalisation, filter = true)

println("========================================")
println("RUN 1: denorm = false (normalized space)")
println("========================================")

m, S, K = multitask(
    X, Y, train, test_e_and_f, 0;
    ζ = ζ,
    normalisation = normalisation,
    denorm = false
)

E_pred = m[1:length(test_features)]
F_pred = m[length(test_features)+1:end]
Var = diag(S)

_, mean, std = select_observations(X, Y, train, true)
truth, _, _  = select_observations(X, Y, test_e, true, mean, std)
forces, _, _ = select_observations(X, Y, test_f, false)

forces = forces ./ std

# Need the number of atom to divide by.
numbers_of_atoms = [ size(X[i][3])[1] for i in test_e_and_f.e.p ]            
for s in test_e_and_f.e.s
    append!(numbers_of_atoms, [ size(X[i][3])[1] for i in s ])
end

ε       = r2(E_pred, truth)
εforces = r2(F_pred, forces)


plot_predictions(E_pred .*std, truth .*std, numbers_of_atoms)
plot_predictions(F_pred.*std, forces.*std)
plot_eigenvalues(K)

println("Energy R²  = ", ε)
println("Force  R²  = ", εforces)
fnorm = norm.(forces)

Statistics.mean(norm.(truth .- m[1:length(test_features)]) ./ norm.(truth))
println(norm.(forces .- m[length(test_features)+1:end]) ./ norm.(forces))

println(minimum(fnorm), " ", maximum(fnorm), " ", Statistics.std(fnorm))
D = diag(K)

println(
    "argmin = ", argmin(D),
    ", min diag = ", minimum(D),
    ", argmax = ", argmax(D),
    ", max diag = ", maximum(D),
    ", max eig = ", maximum(eigvals(K))
)
K[length(test_features)+1:end,length(test_features)+1:end]
argmax(K[1:length(test_features),1:length(test_features)])
argmax(K[length(test_features)+1:end,length(test_features)+1:end])
maximum(K[length(test_features)+1:end,length(test_features)+1:end])

ne, _, _ = BlockSizes(Y,train)
length(test_features)
score = D ./ median(D)

idx = findall(score .> 100)

fig = Figure()
ax = Axis(fig[1,1], xlabel="point", ylabel="log10(Kii)")
scatter!(ax, 1:length(D), log10.(D))
scatter!(ax, idx, log10.(D[idx]), markersize=10)
vlines!(ax, ne, linewidth=2)
fig

cond(K)


