
using LinearAlgebra
using Random
using Distributions
using Tullio


# Arguments
# X :      an array of tuples, each corresponding to an atomic system X[i] = (gradient feature, stress feature, base feature) 
# key_a :  elements in the array corresponding to the atoms in the rows of the covariance matrix
# key_b :  elements in the array corresponding to the atoms in the columns of the covariance matrix
# C     : one of the functions below, named according to the quantities they compare

function covariance(X, key_a, key_b; C, ζ = 4)
    # cas totalement vide
    if isempty(key_a) && isempty(key_b)
        return zeros(0, 0)
    end

    if isempty(key_a)
        c = sum(size(C(X[b], X[b]; ζ), 2) for b in key_b)
        return zeros(0, c)
    end

    if isempty(key_b)
        r = sum(size(C(X[a], X[a]; ζ), 1) for a in key_a)
        return zeros(r, 0)
    end

    # cas normal
    return vcat([
        hcat([C(X[a], X[b]; ζ) for b in key_b]...)
        for a in key_a
    ]...)
end


######################################################################################

# Below:
# A is a tuple corresponding to one atomic system with components (gradient feature, stress feature, base feature) 
# B is a tuple corresponding to one atomic system with components (gradient feature, stress feature, base feature)
 

# covariance between global energies of systems A and B based on atomic contributions
energy( A, B ;  ζ=4  ) = sum( ( last(A)*last(B)').^ζ )  # returns 1 dimensional value


# covariance between atomic forces of structures A and B
function force( A, B ; ζ=4  )
    δ  = ζ*(ζ-1)*(last(A) * last(B)').^(ζ-2)

    @tullio t1[a,b,i,j] := last(B)[b,s]*first(A)[a,i,j,s]  # sum over components s of SOAP descriptor
    @tullio t2[a,b,k,m] := last(A)[a,z]*first(B)[b,k,m,z]  # sum over components z of SOAP descriptor
    @tullio K[i,j,k,m]  := δ[a,b]*t1[a,b,i,j]*t2[a,b,k,m]  # sum over atomic descriptor indices a and b

    δ  = ζ*(last(A) * last(B)').^(ζ-1)
    @tullio M[i,j,k,m] := δ[a,b]*first(A)[a,i,j,y]*first(B)[b,k,m,y] # i indexes atoms in A and j indexes their coordinates
                                                                     # k indexes atoms in B and m indexes their coordinates

    return reshape(K, prod(size(K)[1:2]),:) + reshape(M, prod(size(M)[1:2]),:)  # returns 3(# atoms in A) x 3(# atoms in B) 
end

# covariance between virial stresses of structures A and B
function stress( A, B ; ζ=4  )
    δ  = ζ*(ζ-1)*(last(A) * last(B)').^(ζ-2)
    X  = getindex(A,2)
    Y  = getindex(B,2)

    @tullio t1[a,b,i,j] := last(B)[b,s]*first(A)[a,i,j,s] # sum over components s of SOAP descriptor
    @tullio t2[a,b,k,m] := last(A)[a,z]*first(B)[b,k,m,z] # sum over components z of SOAP descriptor
    @tullio K[c,j,p,m]  := δ[a,b]*t1[a,b,i,j]*t2[a,b,k,m] * X[i,a,c] * Y[k,b,p]   # result: 3 x 3 x 3 x 3 object

    δ  = ζ*(last(A) * last(B)').^(ζ-1)
    @tullio M[c,j,p,m] := δ[a,b]*first(A)[a,i,j,y]*first(B)[b,k,m,y]  * X[i,a,c] *  Y[k,b,p] # result: 3 x 3 x 3 x 3 object

    s1,s2,s3,s4 = size(K)
    return (1/4) * (reshape( K, s1*s2,:) + reshape(M, s1*s2,:) ) # return 9 x 9 matrix
end


# covariance between the atomic forces of structure A and the global energy of structure B based on local features
function force_energy( A, B ; ζ=4  )
    δ  = ζ*(last(A) * last(B)').^(ζ-1)
    @tullio K[i,j] := δ[a,b] * last(B)[b,r] * first(A)[a,i,j,r]
    return reshape(K,:,1)
end

# covariance between the virial stresses of structure A and the global energy of structure B based on local features
function stress_energy( A, B ; ζ=4  )
    δ  = ζ*(last(A) * last(B)').^(ζ-1)
    X  = getindex(A,2)
    
    @tullio K[c,j] := δ[a,b] * last(B)[b,r] * first(A)[a,i,j,r] * X[i,a,c]   # result: 3x3 structure 

    return (1/2)* reshape(  K, :, 1) # return 9x1 structure
end

# covariance between the virial stresses of structure A and the atomic forces of structure B
function stress_force( A, B ; ζ=4  )
    δ  = ζ*(ζ-1)*(last(A) * last(B)').^(ζ-2)
    X  = getindex(A,2) 

    @tullio t1[a,b,i,j] := last(B)[b,s]*first(A)[a,i,j,s]
    @tullio t2[a,b,k,m] := last(A)[a,z]*first(B)[b,k,m,z]
    @tullio K[c,j,k,m]  := δ[a,b]*t1[a,b,i,j]*t2[a,b,k,m] * X[i,a,c] # result: 3 x 3 x (# atoms in B) x 3 structure

    δ  = ζ*(last(A) * last(B)').^(ζ-1)
    @tullio M[c,j,k,m] := δ[a,b]*first(A)[a,i,j,y]*first(B)[b,k,m,y]  * X[i,a,c]  # result: 3 x 3 x (# atoms in B) x 3 structure

    s1,s2,s3,s4 = size(K)
    return (1/2) * (reshape( K, s1*s2,:) + reshape(M, s1*s2,:) )  # return 9 x 3(# atoms in B) structure
end 

