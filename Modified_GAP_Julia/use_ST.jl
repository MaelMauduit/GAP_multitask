using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask2.jl")


# ============================================================
# Loading full available datasets
# ============================================================

Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26nobccCart.xyz")
Xtest, Ytest, ltest = read_data("/Datasets/bcc.xyz")
X = vcat(Xtrain26, Xtest)
Y = vcat(Ytrain26, Ytest)
# ============================================================
# Selecting train and validation datasets
# ============================================================

# idx = shuffle(1:ltrain26)

# train_features = idx[1:(1*ltrain26) ÷ 8]
# test_features = idx[((1*ltrain26) ÷ 8) + 1:end]

train_features = 1:ltrain26+1
test_features = ltrain26+1:ltrain26+ltest

train = (e = (p = train_features, s = []), f = (p = [], s = []), v = (p = [], s = []))
# train = (e = (p = union(train_features, test_features[20:25]), s = []), f = (p = [], s = []), v = (p = [], s = []))
# train = (e = (p = union(bcc_features, test_features[1]), s = []), f = (p = [], s = []), v = (p = [], s = []))

test_e = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []) )
test_f = (e = (p = [], s = []), f = (p = test_features, s = []), v = (p = [], s = []) )
test_e_and_f = (e = (p = test_features, s = []), f = (p = test_features, s = []), v = (p = [], s = []) )
# ============================================================
# Selecting parameters of the regression
# ============================================================

ζ=4                   # Degree of the polynomial kernel
nS = 0                 # Number of secondary tasks
normalisation = true   # Normalisation of outputs

# ============================================================
# Running the regression
# ============================================================

println("========================================")
println("RUN 1: denorm = false (normalized space)")
println("========================================")

training_points, mean, std = select_observations(X, Y, train, true)

σ² = ( e=( p=8.34, s=[] ), f=( p=8.34, s=[] ), v=( p=8.34, s=[] ))
η  = ( e=( p=1e-3, s=[] ), f=( p=0.1, s=[] ), v=( p=0.01, s=[] ))
ϱ  = ( e=[], f=[], v=[] ) 

K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom = train_model(X, Y, train, 0; estimator = "nll", ζ=4, 
                normalisation=normalisation)

m, S, K = multitask(X, train, test_e_and_f, K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom; ζ=4, normalisation=normalisation)

V = sqrt.(diag(S)) 


# ============================================================
# Studying the results
# ============================================================
truth, _, _  = select_observations(X, Y, test_e, false)


std=1
truth = truth .*std

forces, _, _ = select_observations(X, Y, test_f, false)
m[length(test_features)+5]
E_pred = m[1:length(test_features)] .*std
std_E = V[1:length(test_features)] .*std 
F_pred = m[length(test_features)+1:end] .*std
std_F = V[length(test_features)+1:end] .*std

# Need the number of atoms to normalize the contribution.
nb = numbers_of_atoms_energy(X, test_e)

nb.p
ε       = rmse(E_pred, truth, nb.p)
εforces = rmse(F_pred, forces)

println("Energy RMSE  = ", ε)
println("Force  RMSE  = ", εforces)

y_true = truth ./ nb.p
y_pred = E_pred ./ nb.p

include( path*"/multitask2.jl")

fig = plot_predictions_pro(y_pred, y_true, std_E  ./ nb.p, F_pred, forces, std_F; model_name = "GNN-v2", save_path = "bcc_prediction_energies_only_calibr.png")
fig

    