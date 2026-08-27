using Statistics, MultivariateStats, LinearAlgebra, Distributions, GLMakie
using DataFrames, PrettyTables, Random

path = @__DIR__
include(path*"/multitask2.jl")

Random.seed!(2026)

# ---------------------- Settings ----------------------
ζ = 4                   # Degree of the polynomial kernel
nS = 0                  # Number of secondary tasks
normalisation = true    # Normalisation of outputs
Kfold = 5                # Number of CV folds

rcut = [5.0, 8.0, 10.0]
nmax = [8, 10, 12]
lmax = [6, 8, 10]

# NOTE: if multiple rows in Xtr/Ytr belong to the same structure (shared
# energy), do a GROUP k-fold on structure id instead of plain row index,
# or you risk train/val leakage. Swap make_folds for a grouped version if
# read_data exposes a structure/frame id.
function make_folds(n::Int, K::Int; shuffle=true)
    idx = collect(1:n)
    shuffle && Random.shuffle!(idx)
    sizes = fill(div(n, K), K)
    for i in 1:(n % K); sizes[i] += 1; end
    folds, start = Vector{Vector{Int}}(undef, K), 1
    for k in 1:K
        folds[k] = idx[start:start+sizes[k]-1]
        start += sizes[k]
    end
    folds
end

# ---------------------- K-fold CV grid search ----------------------
res_cv = zeros(Float64, length(rcut), length(nmax), length(lmax), 4)  # meanE, stdE, meanF, stdF

for r in 1:length(rcut), n in 1:length(nmax), l in 1:length(lmax)

    Xtr, Ytr, ltr = read_data("/Datasets/train26all.xyz";
                               r_cut=rcut[r], n_max=nmax[n], l_max=lmax[l], sigma=0.5)

    folds = make_folds(ltr, Kfold)
    εE_folds, εF_folds = Float64[], Float64[]

    for k in 1:Kfold
        val_idx   = folds[k]
        train_idx = vcat(folds[setdiff(1:Kfold, k)]...)

        train_k = (e = (p = train_idx, s = []), f = (p = [], s = []), v = (p = [], s = []))
        val_e   = (e = (p = val_idx,   s = []), f = (p = [],        s = []), v = (p = [], s = []))
        val_f   = (e = (p = [],        s = []), f = (p = val_idx,   s = []), v = (p = [], s = []))
        val_ef  = (e = (p = val_idx,   s = []), f = (p = val_idx,   s = []), v = (p = [], s = []))

        true_E, _, _ = select_observations(Xtr, Ytr, val_e, false)
        nbE = numbers_of_atoms_energy(Xtr, val_e)
        true_F, _, _ = select_observations(Xtr, Ytr, val_f, false)
        
        Kmat, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom =
            train_model(Xtr, Ytr, train_k, 0; estimator="nll", ζ=ζ, normalisation=normalisation)

        m, _, _ = multitask(Xtr, train_k, val_ef, Kmat, kept_lines, data, σ², ϱ, η,
                             mean_for_1_atom, std_for_1_atom; ζ=ζ, normalisation=normalisation)

        pred_E = m[1:length(val_idx)]
        pred_F = m[length(val_idx)+1:end]

        push!(εE_folds, rmse(pred_E, true_E, nbE.p))
        push!(εF_folds, rmse(pred_F, true_F))
    end

    res_cv[r, n, l, :] = [mean(εE_folds), std(εE_folds), mean(εF_folds), std(εF_folds)]

    @info "r_cut=$(rcut[r]) n_max=$(nmax[n]) l_max=$(lmax[l]) → " *
          "RMSE_E=$(round(mean(εE_folds),digits=4))±$(round(std(εE_folds),digits=4))  " *
          "RMSE_F=$(round(mean(εF_folds),digits=4))±$(round(std(εF_folds),digits=4))"
end



rows = NamedTuple[]
for r in 1:length(rcut), n in 1:length(nmax), l in 1:length(lmax)
    push!(rows, (r_cut=rcut[r], n_max=nmax[n], l_max=lmax[l],
                 RMSE_E=res_cv[r,n,l,1], RMSE_F=res_cv[r,n,l,3]))
end
df = DataFrame(rows)

for r in sort(unique(df.r_cut))
    sub = filter(row -> row.r_cut == r, df)
    # tri "naturel" pour une table lisible (grille n_max x l_max)
    sort!(sub, [:n_max, :l_max])

    # meilleur point : RMSE_E minimal, en cas d'égalité RMSE_F minimal
    best_idx = argmin(collect(zip(sub.RMSE_E, sub.RMSE_F)))

    fname = "cv_table_rcut_$(replace(string(r), "." => "_")).tex"
    open(fname, "w") do io
        println(io, "\\begin{tabular}{cccc}")
        println(io, "\\toprule")
        println(io, "\$n_{max}\$ & \$l_{max}\$ & RMSE\$_E\$ (Ha/atom) & RMSE\$_F\$ (Ha/Bohr) \\\\")
        println(io, "\\midrule")
        for (i, row) in enumerate(eachrow(sub))
            e_str = @sprintf("%.5f", row.RMSE_E)
            f_str = @sprintf("%.4f", row.RMSE_F)
            if i == best_idx
                println(io, "\\textbf{$(row.n_max)} & \\textbf{$(row.l_max)} & \\textbf{\$$(e_str)\$} & \\textbf{\$$(f_str)\$} \\\\")
            else
                println(io, "$(row.n_max) & $(row.l_max) & \$$(e_str)\$ & \$$(f_str)\$ \\\\")
            end
        end
        println(io, "\\midrule")
        println(io, "\\textit{Mean} & & \\textit{\$$(@sprintf("%.5f", mean(sub.RMSE_E)))\$} & \\textit{\$$(@sprintf("%.4f", mean(sub.RMSE_F)))\$} \\\\")
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end