println("Starting gradient check")
include("multitask.jl")
include("Opti_all_in_one.jl")
include("tools.jl")

# read small dataset
X, Y, l = read_data("/bcc26.xyz")
println("Loaded ", l, " configs")

# small train/test
train = (e = (p = 1:3, s = []), f = (p = 1:3, s = []), v = (p = [], s = []))
nS = 0
ζ = 2

Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train, ζ)
C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
siz = BlockSizes(Y, train)

# get observations (normalisation = true)
y, mean, std = select_observations(X, Y, train, true)
println("obs len = ", length(y), ", mean = ", mean, ", std = ", std)

lh0 = [0.0, -6.0, -6.0, -6.0]

check_gradient_ad(C, y, siz, nS, lh0)
