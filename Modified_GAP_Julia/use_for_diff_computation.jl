path  = @__DIR__
include( path*"/multitask.jl")
include( path*"/linear_reg.jl")


file18    = path*"/bcc18_log_diff.xyz"
file26    = path*"/bcc26_log_diff.xyz"

configs18 = ase.read( file18, index=":")
configs26 = ase.read( file26, index=":")

l18 = length(configs18)
l26 = length(configs26)
println("len18=", l18)
println("len26=", l26)

# construct features
desc = SOAPDescriptor(species=["Si"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5)
 @time X18 = [ stress_describe( c, desc ) for c in configs18 ]
 @time X26 = [ stress_describe( c, desc ) for c in configs26 ]
# @time X = [ grad_describe( c, desc ) for c in configs ]
println("X18 computed")
println("X26 computed")

Y18 = [ ( e=c.info["dft_energy"],
          f=haskey(c.arrays, "dft_force")  ? reshape(c.arrays["dft_force"],:,1)  : zeros(0,1),
          id=haskey(c.info, "id")  ? [c.info["id"]]  : Int[],
          v=haskey(c.info,   "dft_virial") ? reshape(c.info["dft_virial"],:,1)   : zeros(0,1) )
for c in configs18 ]

Y26 = [ ( e=c.info["dft_energy"],
          f=haskey(c.arrays, "dft_force")  ? reshape(c.arrays["dft_force"],:,1)  : zeros(0,1),
          id=haskey(c.info, "id")  ? [c.info["id"]]  : Int[],
          v=haskey(c.info,   "dft_virial") ? reshape(c.info["dft_virial"],:,1)   : zeros(0,1) )
for c in configs26 ]

ϱs18 = linear_regression(Y26,Y18)

X = vcat(X26, X18)
Y = vcat(Y26, Y18)


train = (
    e = (
        p = 1:l26-5,
        s = [[]]
    ),

    f = (
        p = 1:l26-5,
        s = [[]]
    ),

    v = (
        p = [],
        s = [[]]
    )
)
# set training and testing indices (choosing small sets to speed things up)
# train = (
#     e = (
#         p = 1:2:l26,
#         s = [l26+1:2:l26+l18, l26+l18+1:2:l26+l18+l10]
#     ),

#     f = (
#         p = 1:4,
#         s = [l26+1:l26+5, l26+l18+1:l26+l18+5]
#     ),

#     v = (
#         p = [],
#         s = [[],[]]
#     )
# )

# test  = ( e = ( p=2:2:50,  s=[2:2:25,27:2:50]), f= ( p=[3,6,11,13], s=[[4,5,8],[10,11,12]]), v=( p=[3,6,11,13], s=[[4,5,8],[10,11,12]]))

# stack all atoms from all configs → (total_atoms, n_features)
X_all = vcat([x[3] for x in X[train.e.p]]...)

# per-feature mean and std across all atoms and configs
X_mean = mean(X_all, dims=1)   # (1, n_features)
X_std = max.(std(X_all, dims=1), 1e-10)

# normalize: apply to both descriptors and gradients
# dX: mean drops out under derivative, only std remains
X_norm = [
    (
        x[1] ./ reshape(X_std, 1, 1, 1, :),   # dX: divide by std only
        x[2],                                   # R: unchanged
        (x[3] .- X_mean) ./ X_std              # X: center and scale
    )
    for x in X
]


# number of atoms per config (force is 3N × 1)
n_atoms = [size(y.f, 1) ÷ 3 for y in Y[train.e.p]]
n_atoms_tot = [size(y.f, 1) ÷ 3 for y in Y]

# per-atom energy mean → reference energy
e_peratom = [Y[i].e / n_atoms[i] for i in eachindex(Y[train.e.p])]
e_mean = mean(e_peratom)   # eV/atom reference

# std of centered energies → shared scale for e, f, v
e_centered = [Y[i].e - n_atoms[i] * e_mean for i in eachindex(Y[train.e.p])]
e_std = std(e_centered)

# normalize
Y_norm = [
    (
        e  = (Y[i].e - n_atoms_tot[i] * e_mean) / e_std,
        f  = Y[i].f ./ e_std,    # same scale, no mean (constant → 0 under ∂/∂R)
        v  = Y[i].v ./ e_std,    # same scale, no mean
        id = Y[i].id
    )
    for i in eachindex(Y)
]

@time σ²e, ηe, ηf, ηv = optim(X, Y, train, 0, 1, 4., nothing, nothing, true)

@printf(
    "Principal optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
    σ²e, ηe, ηf, ηv
)

@time σ²es1, ηes1, ηfs1, ηvs1 = optim(X_norm, Y_norm, train, 1, 1, 4. , ϱs18, σ²e, true)
@printf(
    "Secondary 1 optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
    σ²es1, ηes1, ηfs1, ηvs1 )

test  = ( e = ( p=l26-4:l26,  s=[[]]), f= ( p=[], s=[[]]), v=( p=[], s=[[]]))

# set hyperparamters (these aren't optimized)
σ² = ( e=( p=900, s=[10] ), f=( p=900, s=[10] ), v=( p=90, s=[10] ))
η  = ( e=( p=1e-5, s=[1e-5] ), f=( p=1e-5, s=[1e-5] ), v=( p=1e-5, s=[1e-5] ))
ϱ  = ( e=[1], f=[1], v=[1] ) #Be careful, rho has to be the same for e,f and v by linearity

# run test 
@time m, S = multitask( X, Y, train, test, σ², ϱ, η )
println("multitask computed")

# m_unnormalize = m * Y_s  #TODO
# evaluate error
truth = select_observations( Y, test )
ε     = r2( m, truth)
println("ε=", ε)
# println("m=", m)

se_σ² = [σ²es1]
se_ϱ  = [ϱs18]
se_ηe = [ηes1]
se_ηf = [ηfs1]
se_ηv = [ηvs1]

σ² = ( e=( p=σ²e, s=copy(se_σ²) ), f=( p=σ²e, s=copy(se_σ²) ), v=( p=σ²e, s=copy(se_σ²) ))
ϱ  = ( e=copy(se_ϱ), f=copy(se_ϱ), v=copy(se_ϱ) )
η  = ( e=( p=ηe, s=se_ηe ), f=( p=ηf, s=se_ηf ), v=( p=ηv, s=se_ηv ))

# run test 
@time m, S = multitask( X, Y, train, test, σ², ϱ, η)
println("optimised multitask computed")

# evaluate error
truth = select_observations( Y, test )
ε     = r2( m, truth )
println("ε=", ε)
println(m)
print(truth)
print(std(truth))
println("m=", maximum(abs.(m-truth)))
