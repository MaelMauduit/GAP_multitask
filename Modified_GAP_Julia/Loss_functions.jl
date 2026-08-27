function nll(lh, Info)
    σ²e = 10.0 ^lh[1]
    nS = Info.nS
    if nS == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+nS]
        ϱ = 10 .^lh[nS+2:2*nS+1]
    end
    ηe = 10.0^lh[2*nS+2]
    ηf = 10.0^lh[2*nS+3]
    ηv = 10.0^lh[2*nS+4]

    h = HyperParams(σ²e, σ², ϱ, ηe, ηf, ηv)  
    try
        Cho = full_matrix(h, Info)
        return 0.5 * dot(Info.y, Cho \ Info.y) + sum(log.(diag(Cho.L)))
    catch
        println("inf detected, opinion rejected")
        println("Current hyperparameters: ", h)
        return Inf
    end
end


function map(lh, Info)
    σ²e = 10.0 ^lh[1]
    nS = Info.nS
    if nS == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+nS]
        ϱ = 10 .^lh[nS+2:2*nS+1]
    end
    ηe = 10.0^lh[2*nS+2]
    ηf = 10.0^lh[2*nS+3]
    ηv = 10.0^lh[2*nS+4]

    h = HyperParams(σ²e, σ², ϱ, ηe, ηf, ηv)  
    try
        Cho = full_matrix(h, Info)
        # print("order of magnitude of first terms : 1) ", 0.5 * dot(Info.y, Cho \ Info.y), " 2) ", sum(log.(diag(Cho.L))), "\n")
        τ = 0.5
        return (0.5 * dot(Info.y, Cho \ Info.y) + sum(log.(diag(Cho.L))) 
                + 0.5 * ((lh[2*nS+2] + 7 ) / τ) ^2
                + 0.5 * ((lh[2*nS+3] + 3 ) / τ) ^2
                + 0.5 * ((lh[2*nS+4] + 3 ) / τ) ^2
                + 0.5 * ((lh[1] - 1 ) / τ) ^2)
                
    catch
        println("inf detected, opinion rejected")
        println("Current hyperparameters: ", h)
        return Inf
    end 
end

function CAL(lh, Info, test_set)
    L = 0
    for α in 0:0.1:1
    end

end








function pnll(lh, Info)
    """
    v is not implemented
    """
    if Info.nS == 0
        vect_empty = []
    else
        vect_empty = [[] for _ in 1:Info.nS]
    end
    trainE = (e = Info.train.e, f = (p = [], s = vect_empty), v = (p = [], s = vect_empty))
    trainF = (e = (p = [], s = vect_empty), f = Info.train.f, v = (p = [], s = vect_empty))
    ne, nf, nv = BlockSizes(Info.Y, Info.train)
    yE = Info.y[1:ne]
    yF = Info.y[ne+1:ne+nf]

    sort_lines = sort(Info.kept)

    true_ne = length(sort_lines[sort_lines .<= ne])
    true_nf = length(sort_lines[(sort_lines .> ne) .& (sort_lines .<= ne + nf)])
    true_nv = length(sort_lines[sort_lines .> ne + nf])

    InfoE = DataInfo(Info.Y,yE,trainE,Info.kept,Info.nS,Info.C)
    InfoF = DataInfo(Info.Y,yF,trainF,Info.kept,Info.nS,Info.C)


    σ²e = 10.0 ^lh[1]
    nS = Info.nS
    if nS == 0
        σ² = []
        ϱ = []
    else 
        σ² = 10 .^lh[2:1+nS]
        ϱ = 10 .^lh[nS+2:2*nS+1]
    end
    ηe = 10.0^lh[2*nS+2]
    ηf = 10.0^lh[2*nS+3]
    ηv = 10.0^lh[2*nS+4]

    h = HyperParams(σ²e, σ², ϱ, ηe, ηf, ηv)  
    try
        ChoE = full_matrix(h, InfoE)
        ChoF = full_matrix(h, InfoF)
        wE = 1/true_ne
        wF = 1/true_nf
        # print("order of magnitude of first terms : 1) ", 0.5 * dot(Info.y, Cho \ Info.y), " 2) ", sum(log.(diag(Cho.L))), "\n")
        return (wE * (0.5 * dot(yE, ChoE \ yE) + sum(log.(diag(ChoE.L))))
                + wF * (0.5 * dot(yF, ChoF \ yF) + sum(log.(diag(ChoF.L)))) )   
    catch
        println("inf detected, opinion rejected")
        println("Current hyperparameters: ", h)
        return Inf
    end
end