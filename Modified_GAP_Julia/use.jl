using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask2.jl")

# ============================================================
# Loading full available datasets
# ============================================================

Xtrain26, Ytrain26, ltrain26 = read_data("/Datasets/train26nobccCart.xyz") ## Careful to the used dataset
Xtest, Ytest, lt = read_data("/Datasets/bcc_dft_26.xyz")

X = vcat(Xtrain26, Xtest)
Y = vcat(Ytrain26, Ytest)


# ============================================================
# Selecting train and validation datasets
# ============================================================

idx = shuffle(1:ltrain26)

train_features = idx[1:(3*ltrain26) ÷ 4]
# test_features = idx[((3*ltrain26) ÷ 4) + 1:end]

# train_features = fps(X[1:ltrain26], (2*ltrain26)//3)
test_features = ltrain26+1:ltrain26+lt

train_features = union(train_features, test_features[3], test_features[7], test_features[17])

train = (e = (p = union(train_features), s = []), f = (p = [], s = []), v = (p = [], s = []))
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

σ² = ( e=( p=16, s=[] ), f=( p=1, s=[] ), v=( p=1, s=[] ))
η  = ( e=( p=1e-8, s=[] ), f=( p=6, s=[] ), v=( p=0.01, s=[] ))
ϱ  = ( e=[], f=[], v=[] ) 

K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom = train_model(X, Y, train, 0; estimator = "nll", ζ=4, normalisation=normalisation)
m, S, K = multitask(X, train, test_e_and_f, K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom; ζ=4, normalisation=normalisation, denorm = false)

V = sqrt.(diag(S)) 


# ============================================================
# Studying the results
# ============================================================
training_points, mean, std = select_observations(X, Y, train, true)
truth, _, _  = select_observations(X, Y, test_e, true, mean, std)
truth = truth .*std

forces, _, _ = select_observations(X, Y, test_f, false)

E_pred = m[1:length(test_features)] .*std
std_E = V[1:length(test_features)] .*std
F_pred = m[length(test_features)+1:end] .*std
std_F = V[length(test_features)+1:end] .*std

# Need the number of atoms to normalize the contribution.
nb = numbers_of_atoms_energy(X, test_e)

# plot_predictions(E_pred, truth, std_E, numbers_of_atoms)
# plot_predictions(F_pred, forces)
# plot_eigenvalues(K)


ε       = rmse(E_pred, truth, nb.p)
εforces = rmse(F_pred, forces)

println("Energy R²  = ", ε)
println("Force  R²  = ", εforces)

y_true = truth 
y_pred = E_pred 

fig = plot_predictions_pro(y_pred, y_true, std_E, F_pred, forces, std_F; model_name = "GNN-v2")
fig

volume = vcat((Y[k].volume for k in test_features)...)

train_plot = (e = (p = union(test_features[3], test_features[7], test_features[17]), s = []), f = (p = [], s = []), v = (p = [], s = []))
nb_train = numbers_of_atoms_energy(X, train_plot)

extra_train = union(test_features[3], test_features[7], test_features[17])

idx_plot_train = findall(x -> x in extra_train, train.e.p)

training_points = training_points[idx_plot_train] .* std
train_volume = vcat((Y[k].volume for k in extra_train)...)

fig2 = plot_result(y_pred, std_E, y_true, volume, nb.p, training_points, train_volume , nb_train.p, ε, savedir="pred_E_V.png")

# save("3.forces.png",fig2)
# save("4.energies.png",fig2)


# training_points = training_points .* std
# train_volume = vcat((Y[k].volume for k in train_features)...)
# numbers_of_atoms_train = [size(X[i][3])[1] for i in train.e.p]
# for s in train.e.s
#     append!(numbers_of_atoms_train, [size(X[i][3])[1] for i in s])
# end
# training_points = training_points[1:1:length(train_features)] ./ numbers_of_atoms_train

# plot_result(y_pred, std_E, y_true, volume, [], [] , ε)



