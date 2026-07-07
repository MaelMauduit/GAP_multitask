path  = @__DIR__
include( path*"/multitask.jl")
include( path*"/linear_reg.jl")
include( path*"/Opti_all_in_one.jl")


file18    = path*"/fcc18_diff.xyz"
file26    = path*"/fcc26_diff.xyz"

configs18 = ase.read( file18, index=":")
configs26 = ase.read( file26, index=":")

l18 = length(configs18)
l26 = length(configs26)

println("len18=", l18)
println("len26=", l26)

# construct features
desc = SOAPDescriptor(species=["Si"], r_cut=20.0, n_max=8, l_max=6, sigma=0.5)

@time X18 = [ stress_describe( c, desc ) for c in configs18 ]
@time X26 = [ stress_describe( c, desc ) for c in configs26 ]
# @time X = [ grad_describe( c, desc ) for c in configs ]

println("X18 computed")
println("X26 computed")

# extract true energies, forces, stresses

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
# plot_regression(Y26,Y10)

X = vcat(X26, X18)
Y = vcat(Y26, Y18)

train = (
    e = (
        p = 1:2:l26,
        # s = [l26+1:l26+l18]
        s = [[l26+1], [l26+1]]
    ),

    f = (
        p = 1:2:l26,
        # s = [l26+1:l26+l18]                
        s = [[l26+1], [l26+1]]

    ),

    v = (
        p = 1:2:l26,
        # s = [l26+1:l26+l18]
        s = [[l26+1], [l26+1]]

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


# @time σ²e, ηe, ηf, ηv = optim(X, Y, train, 0, 2, 2., nothing, nothing, true)

# @printf(
#     "Principal optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
#     σ²e, ηe, ηf, ηv
# )

# @time σ²es1, ηes1, ηfs1, ηvs1 = optim(X, Y, train, 1, 2, 2. , ϱs18, σ²e, true)
# @printf(
#     "Secondary 1 optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
#     σ²es1, ηes1, ηfs1, ηvs1 )

# @time σ²es2, ηes2, ηfs2, ηvs2 = optim(X, Y, train, 2, 2, 2. , ϱs10, σ²e, true)

# @printf(
#     "Secondary 2 optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
#     σ²es2, ηes2, ηfs2, ηvs2 )


Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train)

σ², ϱ, η = Global_optimizer(Y, train, 2, Ce, Cf, Cv, Cfe, Cve, Cvf, false)
σ², ϱ, η = construct_hyperparam([-6.594772516155849, -7.514321333337072, -0.0184713061782979, -6.461803483644921], 1)
# σ², ϱ, η = construct_hyperparam([2.,2.,0.,-6.], 1)
test  = ( e = ( p=1:l26,  s=[[]]), f= ( p=[], s=[[]]), v=( p=[], s=[[]]))

K = construct_covariance(X, train, σ², ϱ, η, true)
println(Diagonal(K))
# # set hyperparamters (these aren't optimized)
# σ² = ( e=( p=900, s=[10,10] ), f=( p=900, s=[10,10] ), v=( p=900, s=[10,10] ))
# η  = ( e=( p=1e-5, s=[1e-5,1e-5] ), f=( p=1e-5, s=[1e-5,1e-5] ), v=( p=1e-5, s=[1e-5,1e-5] ))
# ϱ  = ( e=[1,1], f=[1,1], v=[1,1] ) #Be careful, rho has to be the same for e,f and v by linearity

# # run test 
# @time m, S = multitask( X, Y, train, train, σ², ϱ, η )
# println("multitask computed")

# # evaluate error
# truth = select_observations( Y, train )
# ε     = r2( m, truth)
# println("r2=", ε)
# println("m=", m)

# run test 
normalisation = false
@time m, S = multitask( X, Y, train, test, σ², ϱ, η; normalisation = normalisation)
println(m)
println("optimised multitask computed")
print(σ², ϱ, η)
# evaluate error
truth, meanY, std = select_observations( Y, test, normalisation )
truth = truth .* std .+ meanY 

println(truth)
ε     = r2( m, truth )
println("ε=", ε)

println("m=", maximum(abs.(m-truth)))

# struct CovMatrices
#     Ce::Any
#     Cf::Any
#     Cv::Any
#     Cfe::Any
#     Cve::Any
#     Cvf::Any
# end

# C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)

# number_of_tasks = length(Ce.ps)
# vect_one = ones(number_of_tasks)

# siz = BlockSizes(Y, train)

# y = select_observations(Y, train, true)[1]
# include( path*"/Opti_all_in_one.jl")

# lh0 = vcat([2.0], fill(2.0,1), fill(0., 1), [-3.0])

# G, Gad = check_gradient_ad(C, y, siz, 1, lh0)

