
# ------------------------------------------------------------------------
# SOAP Descriptors using QUIP
#
# QUIP provides analytical gradients for periodic systems with correct
# diagonal terms and O(N) scaling.
# ------------------------------------------------------------------------

using PyCall
using LinearAlgebra
using StaticArrays
using NeighbourLists



# Lazy-loaded Python modules
const _quippy_descriptors = PyNULL()
const _ase_data = PyNULL()

function _init_quip()
    if ispynull(_quippy_descriptors)
        copy!(_quippy_descriptors, pyimport("quippy.descriptors"))
    end
end

function _init_ase_data()
    if ispynull(_ase_data)
        copy!(_ase_data, pyimport("ase.data"))
    end
end

"""
Convert species symbols to atomic numbers using ASE data.
"""
function species_to_Z(species::Vector{String})
    _init_ase_data()
    return [_ase_data.atomic_numbers[s] for s in species]
end

"""
QUIP-based SOAP descriptor.
"""
struct SOAPDescriptor
    pydesc::PyObject
end

"""
    SOAPDescriptor(; species, r_cut, n_max, l_max, sigma)

Create a SOAP descriptor using QUIP.

# Arguments
- `species`: List of chemical species (e.g., ["Si"])
- `r_cut`: Cutoff radius in Angstroms
- `n_max`: Number of radial basis functions
- `l_max`: Maximum angular momentum
- `sigma`: Gaussian smearing width
"""
function SOAPDescriptor(; species=["Si"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5)
    _init_quip()

    Z = species_to_Z(species)
    n_Z = length(Z)
    Z_str = join(Z, " ")

    desc_str = "soap cutoff=$r_cut l_max=$l_max n_max=$n_max atom_sigma=$sigma n_Z=$n_Z Z={$Z_str}"

    return SOAPDescriptor(_quippy_descriptors.Descriptor(desc_str))
end

# Alias for backwards compatibility
const QUIPDescriptor = SOAPDescriptor  

"""
    describe(config, desc::SOAPDescriptor)

Compute normalized SOAP descriptor (no gradients).

Returns: X with shape (n_atoms, n_features), normalized per atom.
"""
function describe(config, desc::SOAPDescriptor)
    result = desc.pydesc.calc(config)
    X = result["data"]
    m = sqrt.(sum(X.^2; dims=2))
    return X ./ m # Ici tu normalise ton descripteur -> check si c'est pris en compte dans le calcul du kernel
end

"""
    grad_describe(config, desc::SOAPDescriptor)

Compute normalized SOAP descriptor with gradients.

Returns: (dX, X) where:
- dX has shape (n_atoms, n_atoms, 3, n_features) - derivatives
- X has shape (n_atoms, n_features) - normalized descriptors

dX[l, a, c, :] = how descriptor of atom l changes when atom a moves in direction c
"""
function grad_describe(config, desc::SOAPDescriptor)
    result = desc.pydesc.calc(config, grad=true)

    X_raw = result["data"]
    n_atoms = size(X_raw, 1)
    n_features = size(X_raw, 2)

    m = sqrt.(sum(X_raw.^2; dims=2))
    X = X_raw ./ m
    
    # Handle single-atom case
    if n_atoms == 1
        dX = zeros(1, 1, 3, n_features)
        return (dX, X)
    end

    # Get gradient data - handle different QUIP versions
    grad_data_raw = result["grad_data"]
    if ndims(grad_data_raw) == 3
        s = size(grad_data_raw)
        if s[2] == 3
            grad_data = grad_data_raw
        elseif s[3] == 3
            grad_data = permutedims(grad_data_raw, (1, 3, 2))
        else
            error("Unexpected grad_data shape: $s")
        end
    else
        error("Unexpected grad_data dimensions: $(ndims(grad_data_raw))")
    end

    # Get index mapping (0-based from Python)
    grad_idx = result["grad_index_0based"]
    n_pairs = size(grad_idx, 1)

    # Convert sparse to dense, accumulating multiple contributions per pair
    dX_raw = zeros(n_atoms, n_atoms, 3, n_features)
    for i in 1:n_pairs
        central = Int(grad_idx[i, 1]) + 1
        neighbor = Int(grad_idx[i, 2]) + 1
        for c in 1:3
            dX_raw[central, neighbor, c, :] .+= grad_data[i, c, :]
        end
    end

    # Apply normalization chain rule: dX̂/dG = dX/dG / ||X|| - (X · dX/dG) * X / ||X||³

    dX = similar(dX_raw)
    for l in 1:n_atoms, a in 1:n_atoms, c in 1:3
        dX_raw_lac = dX_raw[l, a, c, :]
        X_raw_l = X_raw[l, :]
        m_l = m[l]
        dX[l, a, c, :] = dX_raw_lac / m_l - (dot(dX_raw_lac, X_raw_l)) * X_raw_l / (m_l^3)
    end

    return (dX, X)
end


"""
    stress_describe(config, desc::SOAPDescriptor)

Compute normalized SOAP descriptor with gradients.

Returns: (dX, R, X) where:
- dX has shape (n_atoms, n_atoms, 3, n_features) - derivatives
- R has shape (n_atoms, n_atoms, 3) - distance to neighbors within cutoff
- X has shape (n_atoms, n_features) - normalized descriptors

dX[l, a, c, :] = how descriptor of atom l changes when atom a moves in direction c
"""
function stress_describe(config, desc::SOAPDescriptor)
    result = desc.pydesc.calc(config, grad=true)

    X_raw = result["data"]
    n_atoms = size(X_raw, 1)
    n_features = size(X_raw, 2)

    m = sqrt.(sum(X_raw.^2; dims=2))
    X = X_raw ./ m

    # Handle single-atom case
    if n_atoms == 1
        dX = zeros(1, 1, 3, n_features)
        R  = zeros(1, 1, 3)
        return (dX, R, X)
    end

    # Get gradient data - handle different QUIP versions
    grad_data_raw = result["grad_data"]
    if ndims(grad_data_raw) == 3
        s = size(grad_data_raw)
        if s[2] == 3
            grad_data = grad_data_raw
        elseif s[3] == 3
            grad_data = permutedims(grad_data_raw, (1, 3, 2))
        else
            error("Unexpected grad_data shape: $s")
        end
    else
        error("Unexpected grad_data dimensions: $(ndims(grad_data_raw))")
    end

    # Get index mapping (0-based from Python)
    grad_idx = result["grad_index_0based"]
    n_pairs = size(grad_idx, 1)

    # Convert sparse to dense, accumulating multiple contributions per pair
    dX_raw = zeros(n_atoms, n_atoms, 3, n_features)
    for i in 1:n_pairs
        central = Int(grad_idx[i, 1]) + 1
        neighbor = Int(grad_idx[i, 2]) + 1
        for c in 1:3
            dX_raw[central, neighbor, c, :] .+= grad_data[i, c, :]
        end
    end
        
    # Apply normalization chain rule: dX̂/dG = dX/dG / ||X|| - (X · dX/dG) * X / ||X||³
    dX = similar(dX_raw)
    for l in 1:n_atoms, a in 1:n_atoms, c in 1:3
        dX_raw_lac = dX_raw[l, a, c, :]
        X_raw_l = X_raw[l, :]
        m_l = m[l]
        dX[l, a, c, :] = dX_raw_lac / m_l - (dot(dX_raw_lac, X_raw_l)) * X_raw_l / (m_l^3)
    end

    
    # positions from neighborlist
    cell  = config.get_cell()
    cell  = [ cell[1]'; cell[2]'; cell[3]' ]
    pos   =  config.positions
    pos   =  [ SVector{3,Float64}(pos[i,1],pos[i,2],pos[i,3]) for i in 1:size(pos,1) ]
    pairs = sites( PairList( pos, 5.0, cell, config.pbc))

    R     = zeros( size(pos,1), size(pos,1), 3 )
    for a in 1:size(pos,1)
        inds = getindex.( pairs, 2 )[a] .== a
        R[a,a,:] = sum(getindex.(pairs,3)[a][inds])

        for b in a+1:size(pos,1)
            inds = getindex.( pairs, 2 )[a] .== b
            R[a,b,:] = sum(getindex.(pairs,3)[a][inds])
            R[b,a,:] = -R[a,b,:]
        end
    end

    return ( dX, R, X )

end


