using Statistics
using MultivariateStats
using LinearAlgebra, Distributions
using GLMakie

path  = @__DIR__

include( path*"/multitask copy 2.jl")
include( path*"/Clean_optimisation.jl")
include( path*"/tools.jl")


Xtrain26, Ytrain26, ltrain26 = read_data("/bcc26.xyz") ## Careful to the used dataset
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
    X, Y, train, test_e_and_f, 0, 1;
    ζ = ζ,
    normalisation = normalisation,
    denorm = false
)
m, S, K = multitask(
    X, Y, train, test_e_and_f, 0, 2;
    ζ = ζ,
    normalisation = normalisation,
    denorm = false
)
Var = diag(S)

_, mean, std = select_observations(X, Y, train, true)
truth, _, _  = select_observations(X, Y, test_e, true, mean, std)
forces, _, _ = select_observations(X, Y, test_f, false)

forces = forces ./ std

ε       = r2(m[1:length(test_features)], truth)
εforces = r2(m[length(test_features)+1:end], forces)

println("Energy R²  = ", ε)
println("Force  R²  = ", εforces)


kept_lines = create_filter_cov(X, train, 0)
σ², ϱ, η = Global_optimizer(X, Y, train, kept_lines, 0, normalisation, ζ)
println("The hyperparameters have been optimised with σ² = ", σ², " ϱ = ", ϱ, " η = ", η)

K = construct_covariance(X, train, σ², ϱ, η, true, ζ)
K = K[kept_lines, kept_lines]
cond(K)

D = diag(K)
De = D[1:length(train_features)]
Df = D[length(train_features):end]
Med_e = Statistics.mean(De)
Med_f = Statistics.mean(Df)
frac = sqrt(Med_f / Med_e)

Kprime = copy(K)
Kprime[1:length(train_features), length(train_features)+1:end] = 1/frac .* K[1:length(train_features), length(train_features)+1:end]
Kprime[length(train_features)+1:end, 1:length(train_features)] = 1/frac .* K[length(train_features)+1:end, 1:length(train_features)]
Kprime[length(train_features)+1:end, length(train_features)+1:end] = 1/frac^2 .* K[length(train_features)+1:end, length(train_features)+1:end]
cond(Kprime)

ne, _, _ = BlockSizes(Y, train)

# --- diagonal spread for K ---
score = D ./ median(D)
idx = findall(score .> 100)

# --- diagonal spread for Kprime ---
Dprime = diag(Kprime)
De_p = Dprime[1:length(train_features)]
Df_p = Dprime[length(train_features):end]
score_p = Dprime ./ median(Dprime)
idx_p = findall(score_p .> 100)

# --- conditioning summary ---
println("cond(K)             = ", cond(K))
println("cond(Kprime)        = ", cond(Kprime))
println("improvement factor  = ", cond(K) / cond(Kprime))

println("spread K      : max/min(diag) = ", maximum(D) / minimum(D))
println("spread Kprime : max/min(diag) = ", maximum(Dprime) / minimum(Dprime))

fig2 = Figure(size = (1000, 450))

ax1 = Axis(fig2[1,1], xlabel="point", ylabel="log10(Kii)",
           title="K  (cond = $(round(cond(K), sigdigits=4)))")
scatter!(ax1, 1:length(D), log10.(D))
scatter!(ax1, idx, log10.(D[idx]), markersize=10, color=:red)
vlines!(ax1, ne, linewidth=2, color=:black)

ax2 = Axis(fig2[1,2], xlabel="point", ylabel="log10(K'ii)",
           title="Kprime  (cond = $(round(cond(Kprime), sigdigits=4)))")
scatter!(ax2, 1:length(Dprime), log10.(Dprime))
scatter!(ax2, idx_p, log10.(Dprime[idx_p]), markersize=10, color=:red)
vlines!(ax2, ne, linewidth=2, color=:black)

linkyaxes!(ax1, ax2)
fig2


nS = 0
vect_one = ones(nS)
σ² = (e = (p = 1.0, s = 1. * vect_one), f = (p = 1.0, s = 1. * vect_one), v = (p = 1.0, s = 1.0 * vect_one)) 
η = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
ϱ = (e = 1.0 .* ones(nS), f = 1.0 .* ones(nS), v = 1.0 .* ones(nS))
K_no_hyper = construct_covariance(X, train, σ², ϱ, η, true, ζ)
kept_lines = filter_cov(K_no_hyper)
Kprime = K_no_hyper[kept_lines,kept_lines]
cond(Kprime)


using LinearAlgebra, Statistics, GLMakie

# --- run both configurations, keeping results separate ---
m1, S1, K1 = multitask(
    X, Y, train, test_e_and_f, 0, 1;
    ζ = ζ, normalisation = normalisation, denorm = false
)
m2, S2, K2 = multitask(
    X, Y, train, test_e_and_f, 0, 1;
    ζ = ζ, normalisation = normalisation, denorm = false
)

Var1 = diag(S1)
Var2 = diag(S2)

# --- ground truth (shared across both) ---
_, mean, std = select_observations(X, Y, train, true)
truth, _, _  = select_observations(X, Y, test_e, true, mean, std)
forces, _, _ = select_observations(X, Y, test_f, false)
forces = forces ./ std

# --- R² for each configuration ---
ε1        = r2(m1[1:length(test_features)], truth)
εforces1  = r2(m1[length(test_features)+1:end], forces)

ε2        = r2(m2[1:length(test_features)], truth)
εforces2  = r2(m2[length(test_features)+1:end], forces)

println("--- arg = 1 ---")
println("Energy R²  = ", ε1)
println("Force  R²  = ", εforces1)

println("--- arg = 2 ---")
println("Energy R²  = ", ε2)
println("Force  R²  = ", εforces2)

println("--- Δ (arg2 - arg1) ---")
println("ΔEnergy R² = ", ε2 - ε1)
println("ΔForce  R² = ", εforces2 - εforces1)

# --- quick check on what arg=1 vs arg=2 actually changes structurally ---
println("size(K1) = ", size(K1), "   size(K2) = ", size(K2))
println("cond(K1) = ", cond(K1), "   cond(K2) = ", cond(K2))

# --- calibration check: does predicted std track actual error? ---
resid_e1 = abs.(m1[1:length(test_features)] .- truth)
resid_e2 = abs.(m2[1:length(test_features)] .- truth)
println("corr(|residual|, predicted std) arg1 = ",
        cor(resid_e1, sqrt.(Var1[1:length(test_features)])))
println("corr(|residual|, predicted std) arg2 = ",
        cor(resid_e2, sqrt.(Var2[1:length(test_features)])))

# --- scatter comparison: predicted vs true, energies and forces ---
fig5 = Figure(size = (900, 450))

ax_e = Axis(fig5[1,1], xlabel="true energy", ylabel="predicted energy",
            title="Energy: arg1 (R²=$(round(ε1,digits=4))) vs arg2 (R²=$(round(ε2,digits=4)))")
scatter!(ax_e, truth, m1[1:length(test_features)], label="arg=1", markersize=6)
scatter!(ax_e, truth, m2[1:length(test_features)], label="arg=2", markersize=6)
lines!(ax_e, [minimum(truth), maximum(truth)], [minimum(truth), maximum(truth)],
       color=:black, linestyle=:dash, label="ideal")
axislegend(ax_e, position=:rb)

ax_f = Axis(fig5[1,2], xlabel="true force", ylabel="predicted force",
            title="Forces: arg1 (R²=$(round(εforces1,digits=4))) vs arg2 (R²=$(round(εforces2,digits=4)))")
scatter!(ax_f, forces, m1[length(test_features)+1:end], label="arg=1", markersize=6)
# scatter!(ax_f, forces, m2[length(test_features)+1:end], label="arg=2", markersize=6)
lines!(ax_f, [minimum(forces), maximum(forces)], [minimum(forces), maximum(forces)],
       color=:black, linestyle=:dash, label="ideal")
axislegend(ax_f, position=:rb)

fig5





