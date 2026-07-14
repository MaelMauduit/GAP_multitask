using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask.jl")
include( path*"/optimise.jl")
include( path*"/tools.jl")




# ============================================================
# Loading full available datasets
# ============================================================

Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26.xyz") ## Careful to the used dataset
Xbcc26, Ybcc26, lb26 = read_data("/Datasets/bcc26.xyz") 
Xtest, Ytest, lt = read_data("/Datasets/bcc_dft_26.xyz")

X = vcat(Xtrain26, Xbcc26, Xtest)
Y = vcat(Ytrain26, Ybcc26, Ytest)

indx = (e = (p = union(1:ltrain26+lb26), s = []), f = (p = union([]), s = []), v = (p = union([]), s = []) )
X = X_normalise(X, indx) # Normalise the features




# ============================================================
# Selecting train and validation datasets
# ============================================================

# idx = shuffle(1:ltrain26)

# train_features = idx[1:(3*ltrain26) ÷ 4]
# test_features = idx[((3*ltrain26) ÷ 4) + 1:end]

train_features = fps(X[1:ltrain26], (3*ltrain26)//4)
test_features = setdiff(1:ltrain26, train_features)

train = (e = (p = union(train_features), s = []), v = (p = [], s = []), f = (p = train_features, s = []))
# train = (e = (p = union(bcc_features, test_features[1]), s = []), f = (p = [], s = []), v = (p = [], s = []))

test_e = (e = (p = test_features, s = []), f = (p = [], s = []), v = (p = [], s = []) )
test_f = (e = (p = [], s = []), f = (p = test_features, s = []), v = (p = [], s = []) )
test_e_and_f = (e = (p = test_features, s = []), f = (p = test_features, s = []), v = (p = [], s = []) )




# ============================================================
# Selecting parameters of the regression
# ============================================================

ζ=2                    # Degree of the polynomial kernel
nS = 0                 # Number of secondary tasks
normalisation = true   # Normalisation of outputs




# ============================================================
# Running the regression
# ============================================================

println("========================================")
println("RUN 1: denorm = false (normalized space)")
println("========================================")

m, S, K = multitask(
    X, Y, train, test_e_and_f, 0;
    ζ = ζ,
    normalisation = normalisation,
    denorm = false
)
V = sqrt.(diag(S)) 




# ============================================================
# Studying the results
# ============================================================

_, mean, std = select_observations(X, Y, train, true)
truth, _, _  = select_observations(X, Y, test_e, true, mean, std)
truth = truth .*std

forces, _, _ = select_observations(X, Y, test_f, false)

E_pred = m[1:length(test_features)] .*std
std_E = V[1:length(test_features)] .*std
F_pred = m[length(test_features)+1:end] .*std
std_F = V[length(test_features)+1:end] .*std


# Need the number of atoms to normalize the contribution.
numbers_of_atoms = [ size(X[i][3])[1] for i in test_e_and_f.e.p ]            
for s in test_e_and_f.e.s
    append!(numbers_of_atoms, [ size(X[i][3])[1] for i in s ])
end

# plot_predictions(E_pred, truth, std_E, numbers_of_atoms)
# plot_predictions(F_pred, forces)
# plot_eigenvalues(K)


ε       = r2(E_pred, truth)
εforces = r2(F_pred, forces)

println("Energy R²  = ", ε)
println("Force  R²  = ", εforces)

y_true = truth ./ numbers_of_atoms
y_pred = E_pred ./ numbers_of_atoms

fig = plot_predictions_pro(y_pred, y_true, std_E; model_name = "GNN-v2")


 
