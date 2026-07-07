# ------------------------
# ==============================================================================
# File access and editing
# ==============================================================================
# -----------------------

function makeFile(headers,name)

    df = DataFrame()
    [df[!,h] = Any[]  for h in headers ]
    addToFile(df,name)
end

function addToFile(df::DataFrame,name)

    if isfile(name)
        CSV.write(name, df, append=true )
    else
        CSV.write(name, df )
    end
end

function addToFile(df::AbstractArray,name)
    addToFile( DataFrame( reshape(df,1,:) , :auto  ) , name )
end


# ------------------------
# ==============================================================================
# DataFrame search and editing
# ==============================================================================
# -----------------------

# Description: retrieve rows of dataframe where columns equal some provided value

# Inputs: df        - dataframe to be filtered
#         filCols   - columns to screen ( ie [:color, :shape]   )
#         selectors - if these values appear in columns keep the dataframe row ( ie [ ("green", "circle" ), ("orange", "triangle"  )  ]  )
#         leftover  - if true, return all rows filtered out of the dataframe as a second result

# Note: calling the function with the above examples will return all rows of df where the color is green AND shape is circle OR the color is orange AND the shape is triangle

function multifilter(df, filCols, selectors; leftover=false)
    function filterRules(cols...)::Bool
        (cols) in selectors
    end
    return leftover ? (filter( filCols => filterRules , df ), filter( filCols => !filterRules , df )) : filter( filCols => filterRules , df )
end


# Description: retrieve rows indices of dataframe where a given value appears in a given column

# Input: df  - dataframe to be filtered
#        col - column to screen
#        val - value to look for

function filter_inds( df, col, val  )
    return collect(1:size(df,1))[df[!,col] .=== val]
end


# ------------------------
# ==============================================================================
# Array properties and editing
# ==============================================================================
# -----------------------

r(x)             = Int64(round(x))
ℓ(x)             = length(x)
ej(j;n=3)        = I(n)[:,j]
relu(x)          = maximum([0.,x ])
center(a)        =   a .- mean(a)
vcenter(v)       = [ c .- mean(vcat(v...)) for c in v ]
get_loc(b;v)     = findmin( abs.( v .- b )  )[2]
Σ2σ(Σ)           = sqrt.( diag(Hermitian(Σ)) )
nest(V,d)        = [ V[i:d:end] for i in 1:d ]
split(V,d)       = collect(Iterators.partition(V,d))
H(M ; j=1e-6)    = Hermitian(M) + j*I
abstol(M;ε=1e-6) = M<0 && M>-ε ? 0. : M
block_cat(B)     =  vcat( [  [zeros(size(b,1),size(B[1],2)-size(b,2)) b ]    for b in B ]...)
id( M )          = I(maximum(size(M)))[1:size(M,1),1:size(M,2)]

function unpack(V,n) 
    result = []
    start  = 0
    for i in n
        push!(result, θ[start.+(1:i)])
        start+=i
    end
    return result
end

function squish(V)
    result = []
    for v in V
        if isa(v, AbstractArray) 
            append!(result, squish(v))
        else
            push!(result, v)
        end
    end
    return result
end

# ------------------------
# ==============================================================================
# Error and evaluation
# ==============================================================================
# -----------------------

mae(a,b)  = mean(abs.(a-b))
rmae(a,b) = mean(abs.(a-b))/mean(abs.(b))
rmse(a,b) = sqrt(mean((a-b).^2 ))

function euclid²(A)
   X = A*A'
   return diag(X).+ diag(X)' - 2*X
end
euclid²(A,B) =  sum(A.*A ; dims=2) .+ sum(B.*B ;dims=2)' - 2*A*B'

# ------------------------
# ==============================================================================
# ASE atom interface and relaxation
# ==============================================================================
# -----------------------


function atom_extract(config)
    G    = config.positions
    s    = config.symbols
    cell = config.get_cell()
    cell = [ cell[1]'; cell[2]'; cell[3]' ]
    pbc  = config.pbc   
    return G, s, cell, pbc
end

function atom_create(G,s,cell,pbc)
    a = atoms.Atoms( s, G )
    a.set_cell(  cell )
    #a.set_cell(  cell, scale_atoms=true )
    a.pbc = pbc
    return a
end

function atom_create(G,G_perturb,s,cell,perturb_cell,pbc)
    a = atom_create( s, G, cell, pbc )
    a.set_cell(  perturb_cell, scale_atoms=true )
    a.set_positions(G_perturb)
    return a
end

