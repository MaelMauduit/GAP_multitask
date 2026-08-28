    using LinearAlgebra
    using Random
    using CSV
    using DataFrames
    using Distributions
    using SpecialFunctions
    using GLM
    using Optim
    using Printf
    using Statistics

    # import Python libraries
    using PyCall
    ase   = pyimport("ase.io")
    atoms = pyimport("ase")

    # import local files
    path  = @__DIR__
    include( path*"/auxiliary.jl")
    include( path*"/inner_kernel.jl")
    include( path*"/quip_descriptors.jl")
    include( path*"/optimise2.jl")
    include( path*"/tools.jl")

    # ---------------------------------------------------------------
    # ===============================================================
    #
    # Outer loop
    #
    # ===============================================================
    # ---------------------------------------------------------------

    # shorthand
    # t  :: test
    # p  :: primary
    # s  :: secondary
    # e  :: energy
    # f  :: force
    # v  :: virial stress
    # σ² :: variance
    # η  :: noise
    # ϱ  :: correlation

    #----------------------------------------
    # Train multitask model with full covariance
    #----------------------------------------

    # The arguments to this function are hierarchies of arrays and tuples

    # X :: Array of tuples, each element corresponds to a different atomic system: X[i] = ( force features, stress features, features )
    # X may be the output of grad_describe, stress_describe, or describe (the choice of method determines which elements of X[i] are computed)
    # X is a suitable input to the covariance function

    # Y :: Array of named tuples, each element corresponds to a different atomic system: Y[i] = ( e=<energy of system i>, f=<atomic forces>, v=<virial stresses> )

    # train :: Named tuple, ultimately contains array of indices for elements of X, indicate which systems are used in training
    # train = ( e = ( p=[indices for primary task training] , s=[ [indices for s1],  ..., [indices for sn] ] ),  
    #           f = ( p=[indices for primary task training] , s=[ [indices for s1],  ..., [indices for sn] ] ), 
    #           v = ( p=[indices for primary task training] , s=[ [indices for s1],  ..., [indices for sn] ] ) )

    # test :: Named tuple with the same structure as test, but the elements are indices for test predictions

    # σ² :: Named tuple, ultimately contains the variance hyperparameters for the inner product kernels
    # σ² = ( e = (p=variance for primary task energies, s=[ σ² for s1, ..., σ² for sn] ),
    #        f = (p=variance for primary task forces, s=[ σ² for s1, ..., σ² for sn] ),
    #        v = (p=variance for primary task stresses, s=[ σ² for s1, ..., σ² for sn] ) )

    # ϱ :: Named tuple, contains correlation of secondary tasks with primary task
    # ϱ = ( e=[ ϱ for s1 energies, ..., ϱ for sn ], f=[ ϱ for s1 forces, ..., ϱ for sn ], v=[ ϱ for s1 stresses, ..., ϱ for sn ] ]

    # η :: Named tuple, noise variance
    # η = ( e=noise for training energies, f=noise for training forces, v=noise for training stresses )

function numbers_of_atoms_energy(X, key)
    np = Float64[ size(X[i][3])[1] for i in key.e.p ]
    ns = Vector{Vector{Int}}()
    for s in key.e.s
        push!(ns, [ size(X[i][3])[1] for i in s ])
    end
    return (p = np, s = ns)
end

function numbers_of_atoms_viriel(X, key)
    np = Float64[ size(X[i][3])[1] for i in key.v.p ]
    ns = Vector{Vector{Int}}()
    for s in key.v.s
        push!(ns, [ size(X[i][3])[1] for i in s ])
    end
    return (p = np, s = ns)
end

function train_model(X, Y, train, number_of_task; estimator = "nll", ζ=4, normalisation=false, hyper = nothing)
    kept_lines = create_filter_cov_decoupled(X, Y, train,number_of_task,ζ) 
    print(kept_lines)
    if isnothing(hyper)
        σ², ϱ, η = Global_optimizer(X, Y, train, kept_lines, number_of_task; 
                                    estimator = estimator,
                                    normalisation = normalisation, 
                                    ζ = ζ)
    else
        σ², ϱ, η = hyper
    end
    println("The hyperparameters have been optimised with \n σ² = ", σ², "\n ϱ = ", ϱ, "\n η = ", η)


    K = construct_covariance(X, train, σ², ϱ, η, true, ζ)

    h= HyperParams(σ².e.p, σ².e.s, ϱ.e, η.e.p, η.f.p, η.v.p)
    jitter  = get_jitter(h, Y, train, kept_lines, K) 
    ne, ηe = jitter[1]
    println("The problem ?", jitter)
    nf, ηf = jitter[2]
    nv, ηv = jitter[3]
    
    K = K[kept_lines, kept_lines]

    D = Diagonal(vcat(fill(ηe, ne), fill(ηf, nf), fill(ηv, nv)))

    K = K + D
    
    λ = eigvals(K)
    println("The conditionning number is cond(K) = ", cond(K), "\n")
    println("The smallest eigenvalue is ", minimum(λ), " and the biggest is ", maximum(λ))
    
    data, mean_for_1_atom, std_for_1_atom = select_observations(X, Y, train, normalisation)
    data = data[kept_lines]

    return K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom
end


function multitask(X, train, test, K, kept_lines, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom; ζ=4, normalisation=true, denorm = true)
    Kt  = construct_covariance(X, train, test, σ², ϱ, σ², ϱ, ζ)
    Kt = Kt[kept_lines, :]
    Ktt = construct_covariance(X, test,  σ², ϱ, η, false, ζ)
    
    β = K \ Kt
    μ = β' * data
    Σ = Ktt - β' * Kt

    if !denorm
        return μ, Σ, K
    end

    if normalisation
        nb = numbers_of_atoms_energy(X, test)
        all_n_atoms = vcat(nb.p, nb.s...)
        L = length(all_n_atoms)

        μ[1:L] = [(μ[k]*std_for_1_atom+all_n_atoms[k]*mean_for_1_atom) for k in 1:L] 

        numbers_of_forces = [ size(X[i][3])[1] for i in test.f.p ]            
        for s in test.f.s
            append!(numbers_of_forces, [ size(X[i][3])[1] for i in s ])
        end

        nb_forces = 3 * sum(numbers_of_forces; init=0)
        if nb_forces > 0
            μ[L+1:end] = [μ[L+k] * std_for_1_atom for k in 1:nb_forces]
        end
            
        return μ, Σ .* std_for_1_atom^2, K
    end
            
    return μ, Σ, K
end

function addData(X, Y, train, kept_lines, test, add_train, number_of_task, K, data, σ², ϱ, η, mean_for_1_atom, std_for_1_atom;  ζ=4, normalisation=false, denorm = true)
    kept_lines_S = create_filter_cov_decoupled(X, Y, add_train, number_of_task,ζ) 
    K12 = construct_covariance(X, train, add_train, σ², ϱ, σ², ϱ, ζ)
    K12 = K12[kept_lines, kept_lines_S]
    K22 = construct_covariance(X, add_train,  σ², ϱ, η, true, ζ)
    K22 = K22[kept_lines_S, kept_lines_S]

    data_add, _, _ = select_observations(X, Y, add_train, normalisation, mean_for_1_atom, std_for_1_atom)
    data_add = data_add[kept_lines_S]

    full_data = vcat(data, data_add)

    K_inv = Schur_inversion(K, K12, K22)

    Kt  = construct_covariance(X, train, test, σ², ϱ, σ², ϱ, ζ)
    Kt = Kt[kept_lines, :]
    Kt_add = construct_covariance(X, add_train, test, σ², ϱ, σ², ϱ, ζ)
    Kt_add = Kt_add[kept_lines_S, :]
    full_Kt = vcat(Kt, Kt_add)

    Ktt = construct_covariance(X, test,  σ², ϱ, η, false, ζ)

    β = K_inv * full_Kt
    μ = β' * full_data
    Σ = Ktt - β' * full_Kt

    if !denorm
        return μ, Σ, K
    end

    if normalisation
        nb = numbers_of_atoms_energy(X, test)
        all_n_atoms = vcat(nb.p, nb.s...)
        L = length(all_n_atoms)

        μ[1:L] = [(μ[k]*std_for_1_atom+all_n_atoms[k]*mean_for_1_atom) for k in 1:L] 

        numbers_of_forces = [ size(X[i][3])[1] for i in test.f.p ]            
        for s in test.f.s
            append!(numbers_of_forces, [ size(X[i][3])[1] for i in s ])
        end
        
        nb_forces = 3 * sum(numbers_of_forces; init=0)
        if nb_forces > 0
            μ[L+1:end] = [μ[L+k] * std_for_1_atom for k in 1:nb_forces]
        end
            
        return μ, Σ .* std_for_1_atom^2, K
    end
            
    return μ, Σ, K
end




# ---------------------------------------------------------------
# ===============================================================
#
# Construct covariance components
#
# ===============================================================
# ---------------------------------------------------------------

# use to construct train-train and test-test covariances
# the split in two functions will be helpful to accelerate hyperparameters optimisation
function construct_covariance( X::AbstractArray, key::NamedTuple, σ²::NamedTuple, ϱ::NamedTuple, η::NamedTuple, noise::Bool, ζ = 4)
    Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, key, ζ)
    K = hyperparameters(Ce, Cf, Cv, Cfe, Cve, Cvf, σ², ϱ, η, noise, X, key)
    return K
end

function construct_matrices(X::AbstractArray, key::NamedTuple, ζ = 4)
    # construct base covariances, no hyperparameters
    Ce  = triangular(  X, key.e        ; C=energy, ζ )
    Cf  = triangular(  X, key.f        ; C=force, ζ  )
    Cv  = triangular(  X, key.v        ; C=stress, ζ )
    Cfe = rectangular( X, key.f, key.e ; C=force_energy, ζ  )
    Cve = rectangular( X, key.v, key.e ; C=stress_energy, ζ )
    Cvf = rectangular( X, key.v, key.f ; C=stress_force, ζ  )
    return Ce, Cf, Cv, Cfe, Cve, Cvf
end

function hyperparameters(Ce, Cf, Cv, Cfe, Cve, Cvf, σ²::NamedTuple, ϱ::NamedTuple, η::NamedTuple, noise::Bool, X::AbstractArray, key::NamedTuple)  
    # apply hyperparameters
    Ke  = hyper_tri( Ce,  σ².e, ϱ.e, η.e,  noise, numbers_of_atoms_energy(X, key))
    Kf  = hyper_tri( Cf,  σ².f, ϱ.f, η.f,  noise )
    Kv  = hyper_tri( Cv,  σ².v, ϱ.v, η.v,  noise , numbers_of_atoms_viriel(X, key))

    Kfe = hyper_rec( Cfe, σ².f, ϱ.f, σ².e, ϱ.e   )
    Kve = hyper_rec( Cve, σ².v, ϱ.v, σ².e, ϱ.e   )
    Kvf = hyper_rec( Cvf, σ².v, ϱ.v, σ².f, ϱ.f   )

    return [ Ke Kfe' Kve' ; Kfe Kf Kvf' ; Kve Kvf Kv ]
end

# use to construct train-test covariances
function construct_covariance( X::AbstractArray, key_a::NamedTuple, key_b::NamedTuple, σ²_a::NamedTuple, ϱ_a::NamedTuple, σ²_b::NamedTuple, ϱ_b::NamedTuple, ζ = 4)
    # construct base covariances, no hyperparameters
    Ce  = rectangular( X, key_a.e, key_b.e ; C=energy, ζ )
    Cf  = rectangular( X, key_a.f, key_b.f ; C=force, ζ  )
    Cv  = rectangular( X, key_a.v, key_b.v ; C=stress, ζ )
    Cfe = rectangular( X, key_a.f, key_b.e ; C=force_energy, ζ  )
    Cve = rectangular( X, key_a.v, key_b.e ; C=stress_energy, ζ )
    Cvf = rectangular( X, key_a.v, key_b.f ; C=stress_force, ζ  )
    Cef = rectangular( X, key_b.f, key_a.e ; C=force_energy, ζ  )
    Cev = rectangular( X, key_b.v, key_a.e ; C=stress_energy, ζ )
    Cfv = rectangular( X, key_b.v, key_a.f ; C=stress_force, ζ  )

    # apply hyperparameters
    Ke  = hyper_rec( Ce,  σ²_a.e, ϱ_a.e, σ²_b.e, ϱ_b.e )
    Kf  = hyper_rec( Cf,  σ²_a.f, ϱ_a.f, σ²_b.f, ϱ_b.f )
    Kv  = hyper_rec( Cv,  σ²_a.v, ϱ_a.v, σ²_b.v, ϱ_b.v )
    Kfe = hyper_rec( Cfe, σ²_a.f, ϱ_a.f, σ²_b.e, ϱ_b.e )
    Kve = hyper_rec( Cve, σ²_a.v, ϱ_a.v, σ²_b.e, ϱ_b.e )
    Kvf = hyper_rec( Cvf, σ²_a.v, ϱ_a.v, σ²_b.f, ϱ_b.f )
    Kef = hyper_rec( Cef, σ²_a.f, ϱ_a.f, σ²_b.e, ϱ_b.e )'
    Kev = hyper_rec( Cev, σ²_a.v, ϱ_a.v, σ²_b.e, ϱ_b.e )'
    Kfv = hyper_rec( Cfv, σ²_a.v, ϱ_a.v, σ²_b.f, ϱ_b.f )'

    # return constructed matrix
    return [ Ke Kef Kev ; Kfe Kf Kfv ; Kve Kvf Kv ]
end

#----------------------------------------------------------

# construct multitask covariance for property indicated by  C
# this should be used by symmetric matrices, so only the upper right blocks are constructed
function triangular( X, key ; C, ζ = 4 )
    nS = length(key.s) 

    p  =    covariance( X, key.p,    key.p     ; C, ζ )
    ps = [  covariance( X, key.p,    key.s[i]  ; C, ζ )  for i in 1:nS ]
    s  = [[ covariance( X, key.s[i], key.s[j]  ; C, ζ )  for j in i:nS ]  for i in 1:nS ]

    return ( p=p, ps=ps, s=s )
end

# construct multitask covariance for property indicated by  C
# this should be used by non-symmetric matrices, all blocks are constructed
function rectangular( X, key_a, key_b ; C,  ζ = 4)
    nS  = length(key_a.s)

    p   =    covariance( X, key_a.p,    key_b.p     ; C, ζ )
    ps  = [  covariance( X, key_a.p,    key_b.s[i]  ; C, ζ )  for i in 1:nS ]
    sp  = [  covariance( X, key_a.s[i], key_b.p     ; C, ζ )  for i in 1:nS ]
    s   = [[ covariance( X, key_a.s[i], key_b.s[j]  ; C, ζ )  for j in 1:nS ]  for i in 1:nS ]

    return ( p=p, ps=ps, sp=sp, s=s )
end

#---------------------------------------------

# apply hyperparameters and construct a symmetric matrix from upper triangle blocks
function hyper_tri(  C,  σ², ϱ, η,  noise, nb = nothing)
    nS = length(ϱ)
    if !isnothing(nb)

        p  = σ².p*C.p + Diagonal(noise * η.p .* nb.p)
        if nS == 0
            return Symmetric(p) 
        end

        ps = hcat([  C.ps[i]*ϱ[i]*σ².p    for i in 1:nS ]...)
        s  = block_cat([   hcat([  C.s[i][j-i+1]*( (i==j)*σ².s[i] + ϱ[i]*ϱ[j]*σ².p)  + (i==j ? noise * η.s[i] * Diagonal(nb.s[i]) : 0*η.s[i]*id(C.s[i][j-i+1]))   for j in i:nS]...)     for i in 1:nS])
    
    else
        p  = σ².p*C.p + noise*η.p*I
        if nS == 0
            return Symmetric(p) 
        end

        ps = hcat([  C.ps[i]*ϱ[i]*σ².p    for i in 1:nS ]...)
        s  = block_cat([   hcat([  C.s[i][j-i+1]*( (i==j)*σ².s[i] + ϱ[i]*ϱ[j]*σ².p)  + (i==j)*noise*η.s[i]*id(C.s[i][j-i+1])   for j in i:nS]...)     for i in 1:nS])
    end
    return Symmetric( [p ps ; ps' s] )
end


# apply hyperparameters and construct nonsymmetric matrix
function hyper_rec( C, σ²_a, ϱ_a, σ²_b, ϱ_b )
    nS = length(ϱ_a)

    σ² = sqrt( σ²_a.p * σ²_b.p )

    if nS == 0
        return σ² .*C.p
    end
    
    # ϱ  = ϱ_a .* ϱ_b mystake I guess ?

    ps =         hcat([  C.ps[i]*ϱ_b[i]*σ²    for i in 1:nS ]...)
    sp =         vcat([  C.sp[i]*ϱ_a[i]*σ²    for i in 1:nS ]...)
    s  = vcat([  hcat([  C.s[i][j]*( (i==j)*sqrt(σ²_a.s[i]*σ²_b.s[i]) + ϱ_a[i]*ϱ_b[j]*σ²)     for j in 1:nS]...)     for i in 1:nS]...)

    return [σ²*C.p ps ; sp s]
end

#-----------------------------------------

# extract training obervations from full system information Y
function select_observations( X::AbstractArray, Y::AbstractArray, key::NamedTuple, normalisation::Bool, the_mean = nothing, the_std = nothing)
    energies = [ Y[i].e          for i in key.e.p ] 
    forces   = vcat([ vec(-Y[i].f)  for i in key.f.p ]...)
    virials  = vcat([ vec(-Y[i].v)  for i in key.v.p ]...)

    # tâches secondaires
    for s in key.e.s
        append!(energies, [ Y[i].e      for i in s ])
    end
    for s in key.f.s
        isempty(s) && continue
        forces = vcat(forces, vcat([ vec(-Y[i].f) for i in s ]...))
    end
    for s in key.v.s
        isempty(s) && continue
        virials = vcat(virials, vcat([ vec(-Y[i].v) for i in s ]...))
    end

    if normalisation
        nb = numbers_of_atoms_energy(X, key)
        if isempty(nb.p) && isnothing(the_mean)
            println("No primary task for energy, normalisation is not possible")
            return Float64[energies; forces; virials], nothing, nothing
        end

        all_n_atoms    = vcat(nb.p, nb.s...)
        number_of_simu = length(energies)
        @assert number_of_simu == length(all_n_atoms)

        if isnothing(the_mean)
            n_primary = length(key.e.p)
            N_primary = sum(nb.p)
            mean_for_1_atom = sum(energies[1:length(nb.p)]) / N_primary

            Average_local = [(energies[k] - all_n_atoms[k]*mean_for_1_atom) / all_n_atoms[k]
                            for k in 1:n_primary]
            std_E = Statistics.std(Average_local, corrected=false)

            if std_E == 0
                println("a unique value of energy may engender a failed prediction")
                std_E = 1
            end
        else
            mean_for_1_atom = the_mean
            if isnothing(the_std)
                println("The mean is provided but not the std, normalisation is not possible")
                return Float64[energies; forces; virials], nothing, nothing
            end
            std_E = the_std
        end

        energies = [(energies[k] - all_n_atoms[k]*mean_for_1_atom) / std_E
                    for k in 1:number_of_simu]
        forces  = forces  ./ std_E
        virials = virials ./ std_E

        println("Observations have been normalised")
        return Float64[energies; forces; virials], mean_for_1_atom, std_E
    end

    return Float64[ energies; forces; virials ], nothing, nothing
end