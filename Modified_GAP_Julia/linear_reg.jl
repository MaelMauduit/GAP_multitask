using DataFrames
using GLM

function match_by_id(Ya, Yb)
    dict_a = Dict(only(entry.id) => entry.e for entry in Ya if !isempty(entry.id))
    dict_b = Dict(only(entry.id) => entry.e for entry in Yb if !isempty(entry.id))
    
    common_ids = intersect(keys(dict_a), keys(dict_b))
    
    sorted_ids = sort(collect(common_ids))
    
    E1 = [dict_a[id] for id in sorted_ids]
    E2 = [dict_b[id] for id in sorted_ids]
    
    return E1, E2
end


function pearson_regression(Yp, Ys)
    Ep, Es = match_by_id(Yp, Ys)
    rho = cor(Ep, Es)
    return rho
end

function linear_regression(Yp, Ys)
    Ep, Es = match_by_id(Yp, Ys)
    # println("mean Ep = ", mean(Ep))
    # println("mean Es = ", mean(Es))
    # println("mean diff = ", mean(Es .- Ep))
    df = DataFrame(x = Ep, y = Es)
    model = lm(@formula(y ~ x), df)
    β0 = coef(model)[1]
    β1 = coef(model)[2]
    r2 = r²(model)
    println("β0 = ", β0, "  β1 = ", β1, "  R² = ", r2)
    println("to be sure ", β0/(1-β1))
    return β0, β1
end