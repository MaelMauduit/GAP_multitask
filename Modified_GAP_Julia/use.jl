using Statistics
using MultivariateStats
using LinearAlgebra, Distributions

path  = @__DIR__

include( path*"/multitask.jl")
include( path*"/linear_reg.jl")
include( path*"/Opti_all_in_one.jl")
using GLMakie

# desc = SOAPDescriptor(species=["Si"], r_cut=10.0, n_max=8, l_max=6)#, sigma=0.5)

# function read_data(file)
#     f = path*file
#     configs = ase.read(f, index=":")
#     l = length(configs)
#     println("The curentrly processing file has len ", l)
#     X = [stress_describe( c, desc ) for c in configs ]
#     print("Descriptors computed ")
#     Y = [ ( e=c.info["dft_energy"],
#           f=haskey(c.arrays, "dft_force")  ? reshape(c.arrays["dft_force"],:,1)  : zeros(0,1),
#           id=haskey(c.info, "id")  ? [c.info["id"]]  : Int[],
#           volume=haskey(c.info, "volume")  ? [c.info["volume"]]  : Float64[],
#           v=haskey(c.info,   "dft_virial") ? reshape(c.info["dft_virial"],:,1)   : zeros(0,1) )
#     for c in configs ]
#     return X,Y,l
# end

# # X10, Y10, l10 = read_data("/dataset10.xyz")
# # X18, Y18, l18 = read_data("/dataset18.xyz")
# # X26, Y26, l26 = read_data("/dataset26.xyz")
# Xcurve10, Ycurve10, lcurve10 = read_data("/bcc_dft_10.xyz")
# Xcurve, Ycurve, lcurve = read_data("/bcc_dft_26.xyz")
# # # Xbonus10, Ybonus10, lbonus10 = read_data("/bcc_dft_10.xyz")
# # # Xbonus18, Ybonus18, lbonus18 = read_data("/bcc_dft_18.xyz")

# Xtrain10, Ytrain10, ltrain10 = read_data("/train10.xyz")
# Xtrain, Ytrain, ltrain = read_data("/train.xyz")
# Xtest, Ytest, ltest = read_data("/test.xyz")
# X = vcat(Xtrain, Xtrain10, Xtest, Xcurve10, Xcurve)
# Y = vcat(Ytrain, Ytrain10, Ytest, Ycurve10, Ycurve)

# ltr10 = ltrain + ltrain10
# lbcc = ltr10 + ltest
# lc10 = lbcc + lcurve10
# lc = lc10 + lcurve
# # ϱs10 = linear_regression(Y26,Y10)
# # ϱs18 = linear_regression(Y26,Y18)

# X = vcat(X26, X18, X10, Xtest, Xbonus18, Xbonus10)
# Y = vcat(Y26, Y18, Y10, Ytest, Ybonus18, Ybonus10)

# M = reduce(hcat, (X[k][3]' for k in 1:length(X)))  # (n_features × n_atoms_total)

# pca_model = fit(PCA, M; maxoutdim=20)

# P = projection(pca_model)  # (n_features × 20)
# X2 = [ begin
#     d1, d2, d3, d4 = size(X[k][1])
#     (
#         reshape(reshape(X[k][1], d1*d2*d3, d4) * P, d1, d2, d3, size(P,2)),
#         X[k][2],
#         X[k][3] * P
#     )
# end for k in 1:length(X) ]
# M2 = P' * M

function bartletts_sphericity(data)
    n, p = size(data)
    R = cor(data)   
    det_R = det(R)
    ra = rank(R)
    println("rank = ", ra, " over ", p, " data")
    
    # Calculate Chi-square value
    chi2 = -((n - 1) - (2p + 5)/6) * log(det_R)
    
    # Degrees of freedom
    df = (p * (p - 1)) / 2
    
    # Calculate p-value
    p_value = 1 - cdf(Chisq(df), chi2)
    
    return chi2, df, p_value
end

# bartletts_sphericity(M2)


# K = construct_covariance( X, (e = (p = 1:ltrain, s = []), f = (p = [], s = []), v = (p = [], s = []), ), σ², ϱ, η, true, ζ ) # the final argument applies additive noise
# bartletts_sphericity(K)



function fps(X_features, n_select)
    N = size(X_features, 1)
    selected = [1]
    dists = vec(sum((X_features .- X_features[1:1, :]).^2, dims=2))
    
    for _ in 2:n_select
        i = argmax(dists)
        push!(selected, i)
        new_dists = vec(sum((X_features .- X_features[i:i, :]).^2, dims=2))
        dists = min.(dists, new_dists)
    end
    return selected
end

# construire la matrice de features (une ligne par structure)
# F = reduce(vcat, [Statistics.mean(X[k][3], dims=1) for k in 1:length(X)])
# idx = fps(F, 70)  # sélectionne 70 structures diversifiées

# train = (e = (p = idx, s = []), f = (p = [], s = []), v = (p = [], s = []))

# K = construct_covariance( X, train, σ², ϱ, η, true, ζ ) # the final argument applies additive noise
# bartletts_sphericity(K)


# Step = 10
# idx       = shuffle(1:l26)
# train_idx = idx[1:round(Int, 0.6*l26)]
# test_idx  = idx[round(Int, 0.4*l26)+1:end]
train2 = (e = (p = union(1:2:ltrain), s = [union(ltrain+2:2:ltr10 , lbcc+1 : lc10)]), f = ([], s = [[]]), v = (p = [], s = [[]]), )
test = (e = (p = lc10+1:lc, s = [[]]), f = (p = [], s = [[]]), v = (p = [], s = [[]]), )


t = train2
normalisation = true
include( path*"/multitask.jl")
select_observations(X,Y,train2,false)

ζ=2
nS = 1
# t1 = (e = (p = lc10+1:lc, s = [union(lbcc+1 : lc10)]), f = (p = [], s = [[]]), v = (p = [], s = [[]]), )
# t2 = (e = (p = union(lc10+1 : lc), s = []), f = (p = [], s = [[]]), v = (p = [], s = [[]]), )
# a,_,_=select_observations( X, Y, t1, true)
# b,_,_=select_observations( X, Y, t2, true)
# println(a)
# println(b)
Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, t, ζ)
σ², ϱ, η = Global_optimizer(X, Y, t, nS, Ce, Cf, Cv, Cfe, Cve, Cvf, normalisation)
σ², ϱ, η = construct_hyperparam([5.5263616878061885, 1.4591894230606122, -5.121439061154603, -6.6552935843641645], nS)
# test  = ( e = ( p=1+l26+l18+l10: ltest+l26+l18+l10,  s=[]), f= ( p=[], s=[]), v=( p=[], s=[]))
# E10set = ( e = ( p=[],  s=[lbcc+1:lc10]), f= ( p=[], s=[[]]), v=( p=[], s=[[]]))
# E18set = ( e = ( p=[],  s=[1+l26+l18+l10+ltest:l26+l18+l10+ltest+lbonus10]), f= ( p=[], s=[[]]), v=( p=[], s=[[]]))
# run test 
@time m, S = multitask( X, Y, t, test, σ², ϱ, η; ζ = ζ, normalisation = normalisation) 
Var = diag(S)
println("optimised multitask computed")
# evaluate error
# print(select_observations(X,Y,train,true))
truth, mean, std = select_observations( X, Y, test, true)
E10, _, _ = select_observations(X, Y, E10set, false)
# E18, mean, std = select_observations( Y, E18set, false)
# truth = std .* truth .+ mean
# println(truth)
ε     = r2( m, truth )
println("σ²= ", σ².e.p)
println( "η= ", η.e.p)

println("rmse=", ε)
println("Max var = ", maximum(Var)) 

# println("m=", maximum(abs.(m-truth)))

Volume = [Ycurve[k].volume[1] for k in 1:length(Ycurve)]


function plot_result(m, Var, truth, volume, epsilon, savedir = nothing)
    fig = Figure(size=(800, 500))
    ax  = Axis(fig[1,1],
        xlabel="volume (Å³)",
        ylabel="energy (eV)",
        title="prediction vs DFT"
    )

    # Bande ±σ
    if Var !== nothing
        σ = sqrt.(Var)
        band!(ax, volume, m .- σ, m .+ σ;
            color=(:deepskyblue, 0.25))
    end

    lines!(ax, volume, truth,
           linewidth=2, color=:forestgreen, label="DFT")
    scatter!(ax, volume, truth,
             color=:forestgreen, markersize=8)

    # lines!(ax, volume, E10,
    #        linewidth=2, color=:steelblue, label="E10")

    # Prediction
    lines!(ax, volume, m,
           linewidth=2, color=:deepskyblue, label="Prediction")
    # scatter!(ax, volume[union(1,2:Step:end-2,end-1,end)], m[union(1,2,2:Step:end-2,end-1,end)],
    #          color=:deepskyblue, markersize=12)

    text!(ax,
        0.8, 0.8,
        text = rich("ε = $(round(epsilon, digits=4))"),
        align = (:left, :top),
        space = :relative,
        fontsize = 18
    )

    axislegend(ax)
    display(fig)

    if savedir !== nothing
        save(savedir*".png", fig)
    end

end

b = 1

plot_result(E10[b:end] .- 2 * mean, nothing, truth[b:end], Volume[b:end], ε)

# print(m, truth)
# C   = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
# # siz = BlockSizes(Y, train)
# y, _, _ = select_observations(Y, train, false)
# lh0 = [log10(116.87701656786116), -8]
# siz = BlockSizes(Y, train)
# check_gradient_ad(C, y, siz, 0, lh0)

# nll(lh0, C, y, 0)

# K     = construct_covariance(X, t, σ², ϱ, η, true, ζ)
# K_t   = construct_covariance(X, t, test, σ², ϱ, σ², ϱ, ζ)
# lev   = diag(K_t' * (K \ K_t))  # variance expliquée par le train
# Ktt   = construct_covariance(X, test, σ², ϱ, η, false, ζ)
# prior = diag(Ktt)
# println("prior min/max : ", extrema(prior))
# println("lev   min/max : ", extrema(lev))
# println("post  min/max : ", extrema(prior .- lev))  # doit coïncider avec diag(S)
# println(maximum(truth), minimum(truth))
# print(Statistics.std(truth))
# println(m)
# print(truth)

# plot_nll(X, Y, train2, 1, [2.0990077828968654, -2.514498868788125, 0.0017360092185531248, -6.959609736556238], 1, -1, 3, Ce, Cf, Cv, Cfe, Cve, Cvf, true) 
