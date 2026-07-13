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
    include( path*"/optimise.jl")

    function r2(m, truth)
        ss_res = sum((m .- truth).^2)
        ss_tot = sum((truth .- Statistics.mean(truth)).^2)
        return 1 - ss_res / ss_tot
    end

    rmse(y_true, y_pred) = sqrt(Statistics.mean((y_true .- y_pred).^2))

    rmse_relative(y_true, y_pred) = sqrt(Statistics.mean(((y_true .- y_pred).^2) ./ max.(abs.(y_true), 1e-8).^2))
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

function X_normalise(X, train)
    all_indices = [train.e.p; [i for k in train.e.s for i in k]]
    whole_X = vcat([X[i][3] for i in all_indices]...)
    mean_X = vec(Statistics.mean(whole_X, dims=1))  # (253,)
    std_X  = vec(Statistics.std(whole_X,  dims=1))  # (253,)

    L = []
    for k in 1:length(std_X)
        if std_X[k] == 0.0
            std_X[k] = 1.0
            push!(L, k)
        end
    end
    println("The following std coordinates are 0 and replaced by 1: ", L)

    std_4d = reshape(std_X, 1, 1, 1, :)  # (1,1,1,253) pour broadcaster sur (16,16,3,253)

    Xtilde = [(X[k][1] ./ std_4d, X[k][2], (X[k][3] .- mean_X') ./ std_X') for k in 1:length(X)]
    return Xtilde
end

function multitask(X, Y, train, test, number_of_task; ζ=4, normalisation=false, denorm = true)
    kept_lines = create_filter_cov(X,train,number_of_task)
    σ², ϱ, η = Global_optimizer(X, Y, train, kept_lines, number_of_task, normalisation, ζ)
    println("The hyperparameters have been optimised with σ² = ", σ², " ϱ = ", ϱ, " η = ", η)
    K = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    K = K[kept_lines, kept_lines]
    λ = eigvals(K)
    println("The conditionning number is cond(K) = ", cond(K), "\n")
    println("The smallest eigenvalue is ", minimum(λ), " and the biggest is ", maximum(λ))
    Kt  = construct_covariance(X, train, test, σ², ϱ, σ², ϱ, ζ)
    Kt = Kt[kept_lines, :]
    Ktt = construct_covariance(X, test,  σ², ϱ, η, false, ζ)

    data, mean_for_1_atom, std_for_1_atom = select_observations(X, Y, train, normalisation)
    data = data[kept_lines]
    
    β = K \ Kt
    μ = β' * data
    Σ = Ktt - β' * Kt
    if !denorm
        return μ, Σ, K
    end
    if normalisation
        numbers_of_atoms = [ size(X[i][3])[1] for i in test_e_and_f.e.p ]            
        for s in test_e_and_f.e.s
            append!(numbers_of_atoms, [ size(X[i][3])[1] for i in s ])
        end
        L = length(numbers_of_atoms)
        μ[1:L] = [(μ[k]*std_for_1_atom+numbers_of_atoms[k]*mean_for_1_atom) for k in 1:L] 

        numbers_of_forces = [ size(X[i][3])[1] for i in test_e_and_f.f.p ]            
        for s in test_e_and_f.f.s
            append!(numbers_of_forces, [ size(X[i][3])[1] for i in s ])
        end
        nb_forces = 3 * sum(numbers_of_forces)
        μ[L+1 : end] = [(μ[L + k] .*std_for_1_atom) for k in 1:nb_forces]  
            
        return μ, Σ .* std_for_1_atom^2, K
    end
            
        return μ, Σ, K

end

function multitaskv1(X, Y, train, test, σ², ϱ; ζ=4, normalisation=false, filter = nothing)
    nS = length(ϱ)

    # covariances construites sur Y_norm
    K = construct_covariance(X, train, σ², ϱ, η, true, ζ)
    data, mean_for_1_atom, std_for_1_atom = select_observations(X, Y, train, normalisation)
    co = cond(K)
    
    if co > 1e10
        println("Warning: covariance matrix is ill-conditioned, results may be inaccurate as ", "cond(K) = ", co)

    else
        println("Covariance matrix is well-conditioned, proceeding with predictions", " cond(K) = ", co)
    end

    Ktt = construct_covariance(X, test,  σ², ϱ, η, false, ζ)
    Kt  = construct_covariance(X, train, test, σ², ϱ, σ², ϱ, ζ)

    if filter 
        println("Filtering covariance matrix")
        kept_lines = create_filter_cov(X,train,ϱ)
        K = K[kept_lines, kept_lines]
        Kt = Kt[kept_lines, :]
        data = data[kept_lines]
        println("new cond(K) = ", cond(K))
    end

    ηbis = (e = (p = 0., s = 0. * ones(nS)), f = (p = 0., s = 0. * ones(nS)), v = (p = 0., s = 0. * ones(nS)))
    K_0 = construct_covariance(X, train, σ², ϱ, ηbis, true, ζ)
    h = HyperParams(σ².e.p, σ².e.s, ϱ.e)
    η = get_jitter(h, K_0, Y, train)
    println("Jitter for energies: ", η.e.p, " ", η.e.s)
    println("Jitter for forces: ", η.f.p, " ", η.f.s)
    println("Jitter for stresses: ", η.v.p, " ", η.v.s)

    β = K \ Kt
    μ = β' * data
    Σ = Ktt - β' * Kt

    if normalisation
            numbers_of_atoms = [ size(X[i][3])[1] for i in test.e.p ]            
            for s in test.e.s
                append!(numbers_of_atoms, [ size(X[i][3])[1] for i in s ])
            end
        println(μ[1],)
        μ = [(μ[k]*std_for_1_atom+numbers_of_atoms[k]*mean_for_1_atom) for k in 1:length(μ)] 
        println(μ[1], " ", mean_for_1_atom,  " ",std_for_1_atom,  " ",numbers_of_atoms[1])
        return μ, Σ .* std_for_1_atom^2
    end

    return μ, Σ
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
        K = hyperparameters(Ce, Cf, Cv, Cfe, Cve, Cvf, σ², ϱ, η, noise)
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

    function hyperparameters(Ce, Cf, Cv, Cfe, Cve, Cvf, σ²::NamedTuple, ϱ::NamedTuple, η::NamedTuple, noise::Bool )  
        # apply hyperparameters
        Ke  = hyper_tri( Ce,  σ².e, ϱ.e, η.e,  noise )
        Kf  = hyper_tri( Cf,  σ².f, ϱ.f, η.f,  noise )
        Kv  = hyper_tri( Cv,  σ².v, ϱ.v, η.v,  noise )

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
    function hyper_tri(  C,  σ², ϱ, η,  noise )
        nS = length(ϱ)

        p  = σ².p*C.p + noise*η.p*I
        if nS == 0
            return Symmetric(p) 
        end

        ps = hcat([  C.ps[i]*ϱ[i]*σ².p    for i in 1:nS ]...)
        s  = block_cat([   hcat([  C.s[i][j-i+1]*( (i==j)*σ².s[i] + ϱ[i]*ϱ[j]*σ².p)  + (i==j)*noise*η.s[i]*id(C.s[i][j-i+1])   for j in i:nS]...)     for i in 1:nS])
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
            # Construire numbers_of_atoms complet D'ABORD
            numbers_of_atoms = [ size(X[i][3])[1] for i in key.e.p ]
            for s in key.e.s
                append!(numbers_of_atoms, [ size(X[i][3])[1] for i in s ])
            end
            # Maintenant number_of_simu et numbers_of_atoms sont cohérents
            number_of_simu = length(energies)  # == length(numbers_of_atoms)

            # mean sur primaires seulement (cohérent : N_p atomes, E_p énergies)
            n_primary = length(key.e.p)
            N_primary = sum(numbers_of_atoms[1:n_primary])
            mean_for_1_atom = sum(energies[1:n_primary]) / N_primary

            # std sur primaires seulement
            Average_local = [(energies[k] - numbers_of_atoms[k]*mean_for_1_atom) / numbers_of_atoms[k]
                            for k in 1:n_primary]
            std_E = Statistics.std(Average_local, corrected=false)

            if std_E == 0
                println("a unique value of energy may engender a failed prediction")
                std_E = 1
            end

            if  !isnothing(the_mean)
                mean_for_1_atom = the_mean
            end
            if  !isnothing(the_std)
                std_E = the_std
            end
            # normaliser tout (HF + LF) avec ce mean/std
            energies = [(energies[k] - numbers_of_atoms[k]*mean_for_1_atom) / std_E
                        for k in 1:number_of_simu]
            forces  = forces  ./ std_E
            virials = virials ./ std_E

            println("Observations have been normalised")
            return Float64[energies; forces; virials], mean_for_1_atom, std_E
        end

        return Float64[ energies; forces; virials ], nothing, nothing
    end



    # ---------------------------------------------------------------
    # ===============================================================
    #
    # Example code
    #
    # ===============================================================
    # ---------------------------------------------------------------

    # # read data from xyz file with ase
    # file    = path*"/silicon.xyz"
    # configs = ase.read( file, index=":")[vcat(1,400:450)]
    # println("len=", length(configs))

    # # construct features
    # desc = SOAPDescriptor(species=["Si"], r_cut=10.0, n_max=8, l_max=6, sigma=0.5)
    #  @time X = [ stress_describe( c, desc ) for c in configs ]
    # # @time X = [ grad_describe( c, desc ) for c in configs ]
    # println("X computed")

    # # extract true energies, forces, stresses
    # # Y = [ ( e=c.info["dft_energy"],  f=reshape(c.arrays["dft_force"],:,1), v=reshape(c.info["dft_virial"],:,1) )   for c in configs ]
    # Y = [ ( e=c.info["dft_energy"],
    #           f=haskey(c.arrays, "dft_force")  ? reshape(c.arrays["dft_force"],:,1)  : zeros(0,1),
    #           id=haskey(c.info, "id")  ? [c.info["id"]]  : Int[],
    #           v=haskey(c.info,   "dft_virial") ? reshape(c.info["dft_virial"],:,1)   : zeros(0,1) )
    # for c in configs ]


    # # set training and testing indices (choosing small sets to speed things up)
    # train = (
    #     e = (
    #         p = 1:10,
    #         s = [31:39, 21:28]
    #     ),

    #     f = (
    #         p = 1:5,
    #         s = [11:12, 16:20]
    #     ),

    #     v = (
    #         p = 1:10,
    #         s = [21:23,31:33]
    #     )
    # )
    # test  = ( e = ( p=41:49,  s=[[],[]]), f= ( p=[], s=[[],[]]), v=( p=[], s=[[],[]]))
    # # test  = ( e = ( p=2:2:50,  s=[2:2:25,27:2:50]), f= ( p=[3,6,11,13], s=[[4,5,8],[10,11,12]]), v=( p=[3,6,11,13], s=[[4,5,8],[10,11,12]]))
    # lh0 = vcat([2.0], fill(2.0,2), fill(0., 2), [-3.0])

    # Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, train)

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

    # y = select_observations(Y, train)
    # include( path*"/Opti_all_in_one.jl")

    # G, Gad = check_gradient_ad(C, y, siz, 2, lh0)

    # σ², ϱ, η = Global_optimizer(Y, train, Ce, Cf, Cv, Cfe, Cve, Cvf)
    # #First step : Optimization over the feature space : train of σ².p and η.p

    # include( path*"/Optimisation.jl")

    # function plot_zeta(X, Y, train_dataset)
    #     Sub_dataset = (
    #         e = ( p=train_dataset.e.p, s=[ [], [] ] ),
    #         f = ( p=train_dataset.f.p, s=[ [], [] ] ),
    #         v = ( p=train_dataset.v.p, s=[ [], [] ] )
    #     )
        
    #     y = select_observations(Y, Sub_dataset)
        
    #     ζ_range = range(1, 2, length=10)
    #     nll_values = [begin
    #         Ce, Cf, Cv, Cfe, Cve, Cvf = construct_matrices(X, Sub_dataset, ζ)
    #         C = CovMatrices(Ce, Cf, Cv, Cfe, Cve, Cvf)
    #         nll_log([log10(845.52), log10(1e-6), log10(1e-6), log10(1.0)], C, y, 0, nothing, nothing)
    #     end for ζ in ζ_range]

    #     fig, ax, _ = lines(collect(ζ_range), nll_values)
    #     ax.xlabel = "ζ"
    #     ax.ylabel = "NLL"
    #     display(fig)
    # end

    # # plot_zeta(X, Y, train)

    # @time σ²e, ηe, ηf, ηv = optim(X, Y, train, 0, 2)

    # @printf(
    #     "Principal optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
    #     σ²e, ηe, ηf, ηv
    # )

    # @time σ²es1, ηes1, ηfs1, ηvs1 = optim(X, Y, train, 1, 1, 4. , 0.9, σ²e)
    # @printf(
    #     "Secondary 1 optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
    #     σ²es1, ηes1, ηfs1, ηvs1 )


    # @time σ²es2, ηes2, ηfs2, ηvs2 = optim(X, Y, train, 2, 2, 4. , 1., σ²e)
    # @printf(
    #     "Secondary 2 optimisation:\n  σ²e = %.6f\n  ηe  = %.6f\n  ηf  = %.6f\n  ηv  = %.6f\n",
    #     σ²es2, ηes2, ηfs2, ηvs2 )


    # # set hyperparamters (these aren't optimized)
    # σ² = ( e=( p=1, s=[10,10] ), f=( p=1, s=[10,10] ), v=( p=1, s=[10,10] ))
    # η  = ( e=( p=1e-5, s=[1e-5,1e-5] ), f=( p=1e-5, s=[1e-5,1e-5] ), v=( p=1e-5, s=[1e-5,1e-5] ))
    # ϱ  = ( e=[1,2], f=[1,2], v=[1,2] ) #Be careful, rho has to be the same for e,f and v by linearity

    # # run test 
    # # @time m, S = multitask( X, Y, train, test, σ², ϱ, η, 4. )
    # # println("multitask computed")

    # # # evaluate error
    # # truth = select_observations( Y, test )
    # # ε     = r2( m, truth)
    # # println("ε=", ε)
    # # println("m=", m)

    # se_σ² = [σ²es1, σ²es2]    
    # se_ϱ  = [0.9, 1.0]
    # se_ηe = [ηes1,ηes2]
    # se_ηf = [ηfs1,ηfs2]
    # se_ηv = [ηvs1,ηvs2]

    # σ² = ( e=( p=σ²e, s=copy(se_σ²) ), f=( p=σ²e, s=copy(se_σ²) ), v=( p=σ²e, s=copy(se_σ²) ))
    # ϱ  = ( e=copy(se_ϱ), f=copy(se_ϱ), v=copy(se_ϱ) )
    # η  = ( e=( p=ηe, s=se_ηe ), f=( p=ηf, s=se_ηf ), v=( p=ηv, s=se_ηv ))

    # # run test 
    # @time m, S = multitask( X, Y, train, test, σ², ϱ, η )
    # println("optimised multitask computed")

    # # evaluate error
    # truth = select_observations( Y, test )
    # ε     = r2( m, truth )
    # println("r2=", ε)
    # println("m=", m)
    # print(m - truth)


    ####### Comments
    # An issue with the case n == 1 within descriptors calculations
    # An issue with rho hyperparameter in hyper_rec : Twice the multiplication which makes rho⁴ on the diag and other rho issues
    # Add of flexibility : possible not to give some kind of datas which means a change within covariance computation
    # and a more proper way for not to put secondary task with empty secondary hyperparameter
    # Using several rho and sigma for e,f and v does not really make sense in the framework : it would be a nonsense
    ## as those are fidelity correlations and the kernels are separated
    # Do we rly need eta for s when we already consider a delta term
    # Split of the covariance computation in order to make the optimisation more efficient

    # Remember that sigma_p as a role diferent than sigma_s_i (outputscale and inter fidelity covariance parameter)
    # Considering a mu for the delta gp would make sense, and be more logical as the low fidelity is biaised

