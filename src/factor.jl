
function A(j::Int, k::Int, alpha::Real)

    if mod(j+k,2) !=0
        return 0
    end
    bigfact = BigInt(1)
    for p in 1:k
        numer = BigInt(1)
        denom = BigInt(1)
        if p <= k
            numer = p
        end
        if p <= div(k-j,2)
            denom *= p
        end
        bigfact *= (numer//denom)
    end
    powertwo = 1//(BigInt(2^k))
    rising = BigInt(1)
    prod_idx = alpha
    while prod_idx <= alpha + div(j+k,2)
        rising *= (prod_idx == 0 ? 1 : prod_idx)
        prod_idx += 1
    end
    return (powertwo
        * (1//rising)
        *(alpha==0 ? 2 : (alpha+j))
        *(alpha==0 && j==0 ? (1//2) : 1) # first term halved, see cheb poly wiki
        *bigfact)
end


function get_harmonics(normalizer_table, rj_hyps, ra_hyps, j, d)
    max_length_multi = max_num_multiindices(d, size(normalizer_table,1)-1)

    hyp_harms = zeros(Complex{Float64}, length(rj_hyps), max_length_multi) # pre-allocating
    hyp_harms_a = zeros(Complex{Float64}, length(ra_hyps), max_length_multi) # pre-allocating

    N_k_alpha = gegenbauer_normalizer(d, j)
    # println(gegenbauer_normalizer(d, j))
    multiindices =  get_multiindices(d, j)
    hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
    if d > 2
        hyp_harms_k .= hyperspherical.(rj_hyps, j, permutedims(multiindices), Val(false)) # needs to be normalized
        hyp_harms_k ./= normalizer_table[j+1, 1:length(multiindices)]'
    elseif d == 2
        hyp_harms_k .= hypospherical.(rj_hyps, j, permutedims(multiindices)) # needs to be normalized
    end
    hyp_harms_k =  N_k_alpha * conj(hyp_harms_k)

    hyp_harms_k_a = @view hyp_harms_a[:, 1:length(multiindices)]
    if d > 2 # TODO: abstract away
        hyp_harms_k_a .= hyperspherical.(ra_hyps, j, permutedims(multiindices), Val(false))
        hyp_harms_k_a ./= normalizer_table[j+1, 1:length(multiindices)]'
    elseif d == 2
        hyp_harms_k_a .= hypospherical.(ra_hyps, j, permutedims(multiindices)) # needs to be normalized
    end
    return hyp_harms_k, hyp_harms_k_a
end

function get_orthogonal_radials(d, b, degree)
    r = Sym("r")

    weight = r^(d-1)
    polynomials = [Sym(1)]
    polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,b)))
    for i in 2:(degree+1)
        current = r^(i-1)
        for j in 1:(i-1)
            current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, b))
                    / integrate(weight*polynomials[j]^2, (r, 0, b)))
        end
        norm = sqrt(integrate(weight*current^2, (r,0,b)))
        push!(polynomials, expand(current/norm))
    end
    l_polys = []
    for p in polynomials
        push!(l_polys, lambdify(p, [r]))
    end
    B = zeros(degree+1,degree+1)
    for i in 1:(degree+1)
        for j in 1:(degree+1)
            if j == 1
                B[i,j] = subs(polynomials[i], r=>0)
            else
                B[i,j] = polynomials[i].coeff(r^(j-1))
            end
        end
    end
    Binv = inv(B)
    return l_polys, Binv
end


function get_pole_map_trans_table(degree, d, b, a_vals, pij, Binv)
    alpha = (d//2) - 1

    pole_count_map = Dict()
    pole_count = 0

    trans_table = Dict()
    for j in 0:convert(Int, degree/2) # TODO speed up this is so dumb
        multiindex_length = length(get_multiindices(d, j))
        for new_i in 0:(degree-j)
            for n in j:2:(degree-max(new_i,j))
                trans_table[(j,new_i,n)] = 0
                for h in 1:multiindex_length
                    pole_count += 1
                    pole_count_map[(j,new_i,n,h)] = pole_count
                end
            end
        end
    end

    # TODO change new_i var name

    for j in 0:convert(Int, degree/2)
        multiindex_length = length(get_multiindices(d, j))
        for new_i in 0:(degree-j)
            m0 = max(new_i,j)
            if mod(m0+degree-j,2) != 0
                m0+=1
            end
            for n in j:2:(degree-m0)

                for m in m0:2:(degree-n)

                    # m starts at m0
                    for i in (n+m):2:degree
                        for k3 in j:min(convert(Int, degree/2), min(min(n,m), degree-m))
                            if mod(k3+m, 2) != 0 || mod(k3+n,2) != 0
                                continue
                            end
                            trans_table[(j,new_i,n)] += (
                                    Binv[m+1, new_i+1]
                                    * a_vals[i+1]
                                    * pij[i+1, n+m+1]
                                    * (1-delta(0,i)/2)
                                    * A(j,k3, alpha)
                                    * (-2)^k3
                                    * ((1.0/b)^(n+m))
                                    * multinomial([convert(Int,(n-k3)/2),convert(Int,(m-k3)/2),convert(Int,k3)]...) )
                        end
                    end
                end
            end
        end
    end
    return pole_count_map, trans_table
end

function sandbox(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
    to     = fkt_config.to
    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij    = get_pij_table(degree+1)
    d      = fkt_config.d
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, fkt_config.dct_n)
    end

    l_polys, Binv = get_orthogonal_radials(d, b, degree)

    rj_hyps = cart2hyp.(x_vecs)
    ra_hyps = cart2hyp.(y_vecs)

    normalizer_table = hyper_normalizer_table(d, convert(Int, degree/2))

    pole_count_map, trans_table = get_pole_map_trans_table(degree, d, b, a_vals, pij, Binv) # TODO generate pij, binv, avals in function
    U_mat = zeros(Complex{Float64}, length(x_vecs), length(pole_count_map))
    V_mat = zeros(Complex{Float64}, length(pole_count_map), length(y_vecs))
    println(length(pole_count_map))
    for j in 0:convert(Int, degree/2)
        x_harmonics, y_harmonics = get_harmonics(normalizer_table, rj_hyps, ra_hyps, j, d)
        for new_i in 0:(degree-j)
            for n in j:2:(degree-max(new_i,j))
                for h in 1:size(x_harmonics, 2)
                    for (x_idx, x_vec) in enumerate(x_vecs)
                        U_mat[x_idx, pole_count_map[(j,new_i,n,h)]] = (
                            x_harmonics[x_idx, h]
                            * norm(x_vec)^(n)
                            * trans_table[(j,new_i,n,)])
                    end
                    for (y_idx, y_vec) in enumerate(y_vecs)
                        V_mat[pole_count_map[(j,new_i,n,h)], y_idx] = (
                            y_harmonics[y_idx, h]
                            * l_polys[new_i+1](norm(y_vec)))
                    end
                end
            end
        end
    end
    return U_mat, V_mat
end


function degen_kern_harmonic(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
    return sandbox(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)

    to     = fkt_config.to
    r = Sym("r")

    @timeit to "init" begin

    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij = get_pij_table(degree+1)
    d      = fkt_config.d
    dct_n  = fkt_config.dct_n
    num_points = length(x_vecs)
    dover2 = convert(Int, degree/2)
    pole_count= 4*(binomial(dover2+d-1, d) + binomial(dover2+d, d))
    # pole_count = binomial(convert(Int, floor(degree/2))+d+1,d+1)
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, dct_n)
    end

    trunc_param = degree+1

    weight = r^(d-1)
    polynomials = [Sym(1)]
    polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,b)))
    for i in 2:trunc_param
        current = r^(i-1)
        for j in 1:(i-1)
            current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, b))
                    / integrate(weight*polynomials[j]^2, (r, 0, b)))
        end
        norm = sqrt(integrate(weight*current^2, (r,0,b)))
        push!(polynomials, expand(current/norm))
    end
    l_polys = []
    for p in polynomials
        push!(l_polys, lambdify(p, [r]))
    end
    B = zeros(trunc_param,trunc_param)
    for i in 1:trunc_param
        for j in 1:trunc_param
            if j == 1
                B[i,j] = subs(polynomials[i], r=>0)
            else
                B[i,j] = polynomials[i].coeff(r^(j-1))
            end
        end
    end
    Binv = inv(B)

    transtable = zeros(Float64, degree+1, degree+1, degree+1)
    # Binv[m+1, new_i+1]
    for j in 0:dover2
        for new_i in 0:(degree-j)
        # for m in j:(degree-j)
        #     if mod(m+j, 2) != 0
        #         continue
        #     end
        # for n in j:(degree-m)
            for n in j:(degree-max(new_i,j))
                if mod(j+n, 2) != 0
                    continue
                end
                for m in max(new_i,j):(degree-n)
                    if mod(m+j, 2) != 0
                        continue
                    end
                    # for k3 in j:min(min(m,n), min(degree-m, convert(Int64, dover2)))
                    for k3 in j:min(m,n)
                        if mod(k3+n, 2) != 0
                            continue
                        end
                        for i in (n+m):degree
                            m1 = convert(Int,((n-k3)/2))
                            m2 = convert(Int,((m-k3)/2))
                        transtable[j+1, new_i+1, n+1] += (pij[i+1, n+m+1] # here
                                            * a_vals[i+1]
                                            *Binv[m+1, new_i+1]
                                            * A(j, k3, 1//2)
                                            *(1-delta(0,i)/2) # here
                                            *multinomial([m1,m2,k3]...) #here
                                            *((1.0/b)^(n+m)) #here
                                            *((-2.0)^k3)) #here
                        end
                    end
                end
            end
        end
        # println("degree ",j, " ", svdvals(transtable[j+1, :, :]))
        # println("times ", (binomial(j+d-1, j)-binomial(j+d-3,j-2)))
    end

    U_mat = zeros(Complex{Float64}, num_points,pole_count)
    V_mat = zeros(Complex{Float64}, pole_count,num_points)
    pole_counter=0

    end # timeit init


    rj_hyps = cart2hyp.(x_vecs)
    ra_hyps = cart2hyp.(y_vecs)

    max_length_multi = max_num_multiindices(d, dover2)
    hyp_harms = zeros(Complex{Float64}, length(x_vecs), max_length_multi) # pre-allocating
    hyp_harms_a = zeros(Complex{Float64}, length(y_vecs), max_length_multi) # pre-allocating

    # normalizer_table = squared_hyper_normalizer_table(d, dover2)
    normalizer_table = hyper_normalizer_table(d, dover2)

    for j in 0:dover2
        N_k_alpha = gegenbauer_normalizer(d, j)
        # println(gegenbauer_normalizer(d, j))
        multiindices =  get_multiindices(d, j)
        hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
        if d > 2
            hyp_harms_k .= hyperspherical.(rj_hyps, j, permutedims(multiindices), Val(false)) # needs to be normalized
            hyp_harms_k ./= normalizer_table[j+1, 1:length(multiindices)]'
        elseif d == 2
            hyp_harms_k .= hypospherical.(rj_hyps, j, permutedims(multiindices)) # needs to be normalized
        end
        hyp_harms_k =  N_k_alpha * conj(hyp_harms_k)

        hyp_harms_k_a = @view hyp_harms_a[:, 1:length(multiindices)]
        if d > 2 # TODO: abstract away
            hyp_harms_k_a .= hyperspherical.(ra_hyps, j, permutedims(multiindices), Val(false))
            hyp_harms_k_a ./= normalizer_table[j+1, 1:length(multiindices)]'
        elseif d == 2
            hyp_harms_k_a .= hypospherical.(ra_hyps, j, permutedims(multiindices)) # needs to be normalized
        end


        for h in 1:size(hyp_harms_k, 2)
            for new_i in 0:(degree - j)
            # for m in j:2:(degree-j)
                pole_counter +=1
                for n in j:2:(degree-max(new_i,j))
                    for (x_idx, x_vec) in enumerate(x_vecs)
                        U_mat[x_idx, pole_counter] += ((norm(x_vec)^(n))
                                * transtable[j+1, new_i+1, n+1]
                                * hyp_harms_k[x_idx, h])
                    end
                end
                for (y_idx, y_vec) in enumerate(y_vecs)
                    V_mat[pole_counter, y_idx] += (
                    l_polys[new_i+1](norm(y_vec))
                    # * Binv[m+1, new_i+1]
                    * hyp_harms_k_a[y_idx, h])
                end
            end
        end
    end
    return U_mat, V_mat
#     println(N, R, rstar, r)
#
#     idf = idfact(transpose(V_mat), rtol = 1e-6)
# #     approx_mat[:, idf.sk] .= truth_mat[:, idf.sk]
#     println(size(V_mat,1), " ", size(V_mat, 2))
#     println(length(idf.sk))
#     println(idf.sk)
#     for i in 1:length(idf.sk)
#         println(norm(V_mat[idf.sk[i], :]))
#     end
#
#     return U_mat, V_mat, error
end



function degen_kern_nonharmonic(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
    to     = fkt_config.to
    r = Sym("r")

    @timeit to "init" begin

    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij = get_pij_table(degree+1)
    d      = fkt_config.d
    dct_n  = fkt_config.dct_n
    num_points = length(x_vecs)
    dover2 = convert(Int, degree/2)
    pole_count= 4*(binomial(dover2+d-1, d) + binomial(dover2+d, d))
    # pole_count = binomial(convert(Int, floor(degree/2))+d+1,d+1)
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, dct_n)
    end

    trunc_param = degree+1

    weight = r^(d-1)
    polynomials = [Sym(1)]
    polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,b)))
    for i in 2:trunc_param
        current = r^(i-1)
        for j in 1:(i-1)
            current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, b))
                    / integrate(weight*polynomials[j]^2, (r, 0, b)))
        end
        norm = sqrt(integrate(weight*current^2, (r,0,b)))
        push!(polynomials, expand(current/norm))
    end
    l_polys = []
    for p in polynomials
        push!(l_polys, lambdify(p, [r]))
    end
    B = zeros(trunc_param,trunc_param)
    for i in 1:trunc_param
        for j in 1:trunc_param
            if j == 1
                B[i,j] = subs(polynomials[i], r=>0)
            else
                B[i,j] = polynomials[i].coeff(r^(j-1))
            end
        end
    end
    Binv = inv(B)

    transtable = zeros(Float64, degree+1, degree+1, degree+1)
    # Binv[m+1, new_i+1]
    for j in 0:dover2
        for new_i in 0:(degree-j)
        # for m in j:(degree-j)
        #     if mod(m+j, 2) != 0
        #         continue
        #     end
        # for n in j:(degree-m)
            for n in j:(degree-max(new_i,j))
                if mod(j+n, 2) != 0
                    continue
                end
                for m in max(new_i,j):(degree-n)
                    if mod(m+j, 2) != 0
                        continue
                    end
                    # for k3 in j:min(min(m,n), min(degree-m, convert(Int64, dover2)))
                    for k3 in j:min(m,n)
                        if mod(k3+n, 2) != 0
                            continue
                        end
                        for i in (n+m):degree
                            m1 = convert(Int,((n-k3)/2))
                            m2 = convert(Int,((m-k3)/2))
                        transtable[j+1, new_i+1, n+1] += (pij[i+1, n+m+1] # here
                                            * a_vals[i+1]
                                            *Binv[m+1, new_i+1]
                                            * A(j, k3, 1//2)
                                            *(1-delta(0,i)/2) # here
                                            *multinomial([m1,m2,k3]...) #here
                                            *((1.0/b)^(n+m)) #here
                                            *((-2.0)^k3)) #here
                        end
                    end
                end
            end
        end
        # println("degree ",j, " ", svdvals(transtable[j+1, :, :]))
        # println("times ", (binomial(j+d-1, j)-binomial(j+d-3,j-2)))
    end

    U_mat = zeros(Complex{Float64}, num_points,pole_count)
    V_mat = zeros(Complex{Float64}, pole_count,num_points)
    pole_counter=0

    end # timeit init


    rj_hyps = cart2hyp.(x_vecs)
    ra_hyps = cart2hyp.(y_vecs)

    max_length_multi = max_num_multiindices(d, dover2)
    hyp_harms = zeros(Complex{Float64}, length(x_vecs), max_length_multi) # pre-allocating
    hyp_harms_a = zeros(Complex{Float64}, length(y_vecs), max_length_multi) # pre-allocating

    # normalizer_table = squared_hyper_normalizer_table(d, dover2)
    normalizer_table = hyper_normalizer_table(d, dover2)

    for j in 0:dover2
        N_k_alpha = gegenbauer_normalizer(d, j)
        # println(gegenbauer_normalizer(d, j))
        multiindices =  get_multiindices(d, j)
        hyp_harms_k = @view hyp_harms[:, 1:length(multiindices)]
        if d > 2
            hyp_harms_k .= hyperspherical.(rj_hyps, j, permutedims(multiindices), Val(false)) # needs to be normalized
            hyp_harms_k ./= normalizer_table[j+1, 1:length(multiindices)]'
        elseif d == 2
            hyp_harms_k .= hypospherical.(rj_hyps, j, permutedims(multiindices)) # needs to be normalized
        end
        hyp_harms_k =  N_k_alpha * conj(hyp_harms_k)

        hyp_harms_k_a = @view hyp_harms_a[:, 1:length(multiindices)]
        if d > 2 # TODO: abstract away
            hyp_harms_k_a .= hyperspherical.(ra_hyps, j, permutedims(multiindices), Val(false))
            hyp_harms_k_a ./= normalizer_table[j+1, 1:length(multiindices)]'
        elseif d == 2
            hyp_harms_k_a .= hypospherical.(ra_hyps, j, permutedims(multiindices)) # needs to be normalized
        end


        for h in 1:size(hyp_harms_k, 2)
            for new_i in 0:(degree - j)
            # for m in j:2:(degree-j)
                pole_counter +=1
                for n in j:2:(degree-max(new_i,j))
                    for (x_idx, x_vec) in enumerate(x_vecs)
                        U_mat[x_idx, pole_counter] += ((norm(x_vec)^(n))
                                * transtable[j+1, new_i+1, n+1]
                                * hyp_harms_k[x_idx, h])
                    end
                end
                for (y_idx, y_vec) in enumerate(y_vecs)
                    V_mat[pole_counter, y_idx] += (
                    l_polys[new_i+1](norm(y_vec))
                    # * Binv[m+1, new_i+1]
                    * hyp_harms_k_a[y_idx, h])
                end
            end
        end
    end
    return U_mat, V_mat
#     println(N, R, rstar, r)
#
#     idf = idfact(transpose(V_mat), rtol = 1e-6)
# #     approx_mat[:, idf.sk] .= truth_mat[:, idf.sk]
#     println(size(V_mat,1), " ", size(V_mat, 2))
#     println(length(idf.sk))
#     println(idf.sk)
#     for i in 1:length(idf.sk)
#         println(norm(V_mat[idf.sk[i], :]))
#     end
#
#     return U_mat, V_mat, error
end





########################################################################################
########################################################################################
#################  LAST WORKING SANDBOX  ###############################################
########################################################################################
########################################################################################
########################################################################################



function sandbox2(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
    r = Sym("r")
    to     = fkt_config.to
    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij    = get_pij_table(degree+1)
    d      = fkt_config.d
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, fkt_config.dct_n)
    end
    alpha = (d//2) - 1


    weight = r^(d-1)
    polynomials = [Sym(1)]
    polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,b)))
    for i in 2:degree
        current = r^(i-1)
        for j in 1:(i-1)
            current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, b))
                    / integrate(weight*polynomials[j]^2, (r, 0, b)))
        end
        norm = sqrt(integrate(weight*current^2, (r,0,b)))
        push!(polynomials, expand(current/norm))
    end
    l_polys = []
    for p in polynomials
        push!(l_polys, lambdify(p, [r]))
    end
    B = zeros(degree,degree)
    for i in 1:degree
        for j in 1:degree
            if j == 1
                B[i,j] = subs(polynomials[i], r=>0)
            else
                B[i,j] = polynomials[i].coeff(r^(j-1))
            end
        end
    end
    Binv = inv(B)



    rj_hyps = cart2hyp.(x_vecs)
    ra_hyps = cart2hyp.(y_vecs)

    normalizer_table = hyper_normalizer_table(d, convert(Int, degree/2))

    pole_count_map = Dict()
    pole_count = 0

    trans_table = Dict()
    for j in 0:convert(Int, degree/2) # TODO speed up this is so dumb
        multiindex_length = length(get_multiindices(d, j))
        for m in j:2:(degree-j)
            for n in j:(degree-m)
                trans_table[(j,m,n)] = 0
                for h in 1:multiindex_length
                    pole_count += 1
                    pole_count_map[(j,m,n,h)] = pole_count
                end
            end
        end
    end

    for j in 0:convert(Int, degree/2)
        multiindex_length = length(get_multiindices(d, j))
        for m in j:2:(degree-j)
            for n in j:(degree-m)
                for i in (n+m):2:degree
                    for k3 in j:min(convert(Int, degree/2), min(min(n,m), degree-m))
                        if mod(k3+m, 2) != 0 || mod(k3+n,2) != 0
                            continue
                        end
                        trans_table[(j,m,n)] += (
                                  a_vals[i+1]
                                * pij[i+1, n+m+1]
                                * (1-delta(0,i)/2)
                                * A(j,k3, alpha)
                                * (-2)^k3
                                * ((1.0/b)^(n+m))
                                * multinomial([convert(Int,(n-k3)/2),convert(Int,(m-k3)/2),convert(Int,k3)]...) )
                    end
                end
            end
        end
    end

    U_mat = zeros(Complex{Float64}, length(x_vecs), pole_count)
    V_mat = zeros(Complex{Float64}, pole_count, length(y_vecs))

    for j in 0:convert(Int, degree/2)
        x_harmonics, y_harmonics = get_harmonics(normalizer_table, rj_hyps, ra_hyps, j, d)
        for m in j:2:(degree-j)
            for n in j:(degree-m)
                for h in 1:size(x_harmonics, 2)
                    for (x_idx, x_vec) in enumerate(x_vecs)
                        U_mat[x_idx, pole_count_map[(j,m,n,h)]] = (
                            x_harmonics[x_idx, h]
                            * norm(x_vec)^(n)
                            * trans_table[(j,m,n,)])
                    end
                    for (y_idx, y_vec) in enumerate(y_vecs)
                        V_mat[pole_count_map[(j,m,n,h)], y_idx] = (
                            y_harmonics[y_idx, h]
                            * norm(y_vec)^(m) )

                    end
                end
            end

        end
    end
    return U_mat, V_mat
end
