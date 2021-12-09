# Poly will be vec of length 2p
# multiply will double check at first to make sure nothing bad happens

function multiply_polys(a,b)
    ans = zeros(length(a))
    adeg = 0
    for i in length(a):-1:1
        if a[i] != 0
            adeg=i
            break
        end
    end
    bdeg = 0
    for i in length(b):-1:1
        if b[i] != 0
            bdeg=i
            break
        end
    end
    if adeg+bdeg > length(a)
        println("ERROR DROPPING TERMS") # DO NOT DELETE UNLESS TEMPORARILY FOR TIMING
    end
    for i in 1:length(a)
        for j in 1:i
            # i-1 = j-1 + i-j
            ans[i] += a[j]*b[i-j+1]
        end
    end
    return ans
end

function integrate_poly(p, b)
    tot = 0
    for i in 1:length(p)
        tot += ((b^i)/i)*p[i]
    end
    return tot
end

function evaluate_poly(p, x)
    tot = 0
    for i in 1:length(p)
        tot += p[i]*(x^(i-1))
    end
    return tot
end


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

    # weight = r^(d-1)
    weight = zeros(2*(degree+d+1))
    weight[1] = 1
    # weight[d] = 1

    # polynomials = [Sym(1)]
    first_poly = zeros(2*(degree+d+1))
    first_poly[1] = 1
    polynomials = [first_poly]

    # polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,b)))
    polynomials[1] /= sqrt(
                        integrate_poly(
                            multiply_polys(weight,
                            multiply_polys(polynomials[1],polynomials[1])),
                            b))

    for i in 2:(degree+1)
        # current = r^(i-1)
        current = zeros(2*(degree+d+1))
        current[i] = 1

        for j in 1:(i-1)

            # current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, b))
            #         / integrate(weight*polynomials[j]^2, (r, 0, b)))
            rtothei = zeros(2*(degree+d+1))
            rtothei[i] = 1
            current -= (polynomials[j] *
                        integrate_poly(
                        multiply_polys(weight,
                        multiply_polys(polynomials[j],rtothei)),b)
                        / integrate_poly(
                            multiply_polys(weight,
                            multiply_polys(polynomials[j],polynomials[j])),b))
        end
        # norm = sqrt(integrate(weight*current^2, (r,0,b)))
        norm = sqrt(integrate_poly(multiply_polys(weight,
                    multiply_polys(current, current)), b))

                    # push!(polynomials, expand(current/norm))
        push!(polynomials, current/norm)
    end

    # l_polys = []
    # for p in polynomials
    #     push!(l_polys, lambdify(p, [r]))
    # end
    B = zeros(degree+1,degree+1)
    # for i in 1:(degree+1)
    #     for j in 1:(degree+1)
    #         if j == 1
    #             B[i,j] = subs(polynomials[i], r=>0)
    #         else
    #             B[i,j] = polynomials[i].coeff(r^(j-1))
    #         end
    #     end
    # end
    for i in 1:(degree+1)
        for j in 1:(degree+1)
            B[i,j] = polynomials[i][j]
        end
    end
    Binv = inv(B)
    # return polynomials, l_polys, Binv
    return polynomials, Binv
end


function get_pole_map_trans_table(degree, d, b, a_vals, pij, Binv)
    alpha = (d//2) - 1

    M = 2degree
    pole_count_map = Dict()
    pole_count = 0

    trans_table = Dict()
    for harmonic_deg in 0:convert(Int, degree/2) # TODO speed up this is so dumb
        multiindex_length = length(get_multiindices(d, harmonic_deg))
        # for orth_poly_idx in 0:M
        for orth_poly_idx in 0:(degree-harmonic_deg)
            # for n in harmonic_deg:2:M
            for n in harmonic_deg:2:(degree-max(orth_poly_idx,harmonic_deg))
                trans_table[(harmonic_deg,orth_poly_idx,n)] = 0
            end
        end

        for harmonic_ord in 1:multiindex_length
            # for orth_poly_idx in 0:M
            for orth_poly_idx in 0:(degree-harmonic_deg)
                pole_count += 1
                pole_count_map[(harmonic_deg,orth_poly_idx,harmonic_ord)] = pole_count
            end
        end
    end

    for harmonic_deg in 0:convert(Int, degree/2)
        multiindex_length = length(get_multiindices(d, harmonic_deg))
        # for orth_poly_idx in 0:M
        for orth_poly_idx in 0:(degree-harmonic_deg)
            m0 = max(orth_poly_idx,harmonic_deg)
            if mod(m0+degree-harmonic_deg,2) != 0
                m0+=1
            end
            # for n in harmonic_deg:2:M
            for n in harmonic_deg:2:(degree-m0)
                # for m in m0:2:M
                for m in m0:2:(degree-n)
                    for i in (n+m):2:degree
                        for k3 in harmonic_deg:min(convert(Int, degree/2), min(min(n,m), degree-m))
                            if mod(k3+m, 2) != 0 || mod(k3+n,2) != 0
                                continue
                            end
                            trans_table[(harmonic_deg,orth_poly_idx,n)] += (
                                    Binv[m+1, orth_poly_idx+1]
                                    * a_vals[i+1]
                                    * pij[i+1, n+m+1]
                                    * (1-delta(0,i)/2)
                                    * A(harmonic_deg,k3, alpha)
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
    r = Sym("r")

    to     = fkt_config.to
    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij    = get_pij_table(degree+1)
    d      = fkt_config.d
    rtol      = fkt_config.rtol
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, fkt_config.dct_n)
    end

    @timeit to "get ortho radials" begin
    # polynomials, l_polys, Binv = get_orthogonal_radials(d, b, degree)
    polynomials, Binv = get_orthogonal_radials(d, b, degree)
    # polynomials, Binv = get_orthogonal_radials(d, b, 2degree)
    end

    M = 2degree
    rj_hyps = cart2hyp.(x_vecs)
    ra_hyps = cart2hyp.(y_vecs)

    normalizer_table = hyper_normalizer_table(d, convert(Int, degree/2))
    @timeit to "populate umat vmat" begin
    pole_count_map, trans_table = get_pole_map_trans_table(degree, d, b, a_vals, pij, Binv) # TODO generate pij, binv, avals in function
    U_mat = zeros(Complex{Float64}, length(x_vecs), length(pole_count_map))
    V_mat = zeros(Complex{Float64}, length(pole_count_map), length(y_vecs))
    for harmonic_deg in 0:convert(Int, degree/2)
        x_harmonics, y_harmonics = get_harmonics(normalizer_table, rj_hyps, ra_hyps, harmonic_deg, d)
        mis = get_multiindices(d, harmonic_deg)
        for harmonic_ord in 1:size(x_harmonics, 2)
            possibledouble=1
            if d > 2 && mis[harmonic_ord][end] > 0
                possibledouble = 2
            end
            # for orth_poly_idx in 0:M
            for orth_poly_idx in 0:(degree-harmonic_deg)
                for (x_idx, x_vec) in enumerate(x_vecs)
                    # for n in harmonic_deg:2:M
                    for n in harmonic_deg:2:(degree-max(orth_poly_idx,harmonic_deg))
                        U_mat[x_idx, pole_count_map[(harmonic_deg,orth_poly_idx,harmonic_ord)]] += (
                            x_harmonics[x_idx, harmonic_ord]
                            * norm(x_vec)^(n)
                            * trans_table[(harmonic_deg,orth_poly_idx,n)]) * possibledouble
                    end
                end
                for (y_idx, y_vec) in enumerate(y_vecs)
                    # V_mat[pole_count_map[(harmonic_deg,orth_poly_idx,harmonic_ord)], y_idx] = (
                    #     y_harmonics[y_idx, harmonic_ord]
                    #     * l_polys[orth_poly_idx+1](norm(y_vec)))
                    V_mat[pole_count_map[(harmonic_deg,orth_poly_idx,harmonic_ord)], y_idx] = (
                        y_harmonics[y_idx, harmonic_ord]
                        * evaluate_poly(polynomials[orth_poly_idx+1], norm(y_vec)))
                end

            end
        end
    end
    end

    U_polys = Dict()
    V_polys = Dict()
    x = Sym("x")
    @timeit to "form upolys vpolys" begin
    for harmonic_deg in 0:convert(Int, degree/2)
        # for orth_poly_idx in 0:M
        for orth_poly_idx in 0:(degree-harmonic_deg)
            # x_poly = 0
            x_poly = zeros(2*(degree+d+1))
            # x_poly = zeros(2*(2degree+d+1))
            # for n in harmonic_deg:2:M
            for n in harmonic_deg:2:(degree-max(orth_poly_idx,harmonic_deg))
                # x_poly += (x^n)*trans_table[(harmonic_deg,orth_poly_idx,n)]
                xtothenp1 = zeros(2*(degree+d+1))
                # xtothenp1 = zeros(2*(2degree+d+1))
                xtothenp1[n+1] = 1
                x_poly += xtothenp1*trans_table[(harmonic_deg,orth_poly_idx,n)]
            end
            U_polys[(harmonic_deg, orth_poly_idx)] = x_poly
            # V_polys[(harmonic_deg, orth_poly_idx)] = subs(polynomials[orth_poly_idx+1], r=>x)
            V_polys[(harmonic_deg, orth_poly_idx)] = polynomials[orth_poly_idx+1]
        end
    end
    end
    @timeit to "get integrations" begin
    integrations = Dict()
    for harmonic_deg in 0:convert(Int, degree/2)
        N_k_alpha = gegenbauer_normalizer(d, harmonic_deg)
        # for orth_poly_idx1 in 0:M
        for orth_poly_idx1 in 0:(degree-harmonic_deg)
            # for orth_poly_idx2 in 0:M
            for orth_poly_idx2 in 0:(degree-harmonic_deg)
                # prod_poly =  U_polys[(harmonic_deg, orth_poly_idx1)]*V_polys[(harmonic_deg, orth_poly_idx2)]
                prod_poly =  multiply_polys(U_polys[(harmonic_deg, orth_poly_idx1)],V_polys[(harmonic_deg, orth_poly_idx2)])

                # integrations[(harmonic_deg, orth_poly_idx1, orth_poly_idx2)] =  N_k_alpha*integrate(prod_poly*x^(d-1), (x, 0,b))
                xtothed = zeros(2*(degree+d+1))
                # xtothed = zeros(2*(2degree+d+1))
                # xtothed[d] = 1
                xtothed[1] = 1
                integrations[(harmonic_deg, orth_poly_idx1, orth_poly_idx2)] =  (
                      N_k_alpha
                    * integrate_poly(
                        multiply_polys(prod_poly,
                        xtothed), b))
            end
        end
    end
    end
    Cmat = zeros(length(pole_count_map),length(pole_count_map))
    pole_count     = 0
    integral_count = 0
    @timeit to "form cmat" begin
    for harmonic_deg in 0:convert(Int, degree/2)
        multiindex_length = length(get_multiindices(d, harmonic_deg))
        for harmonic_ord in 1:multiindex_length
            # for orth_poly_idx in 0:M
            for orth_poly_idx in 0:(degree-harmonic_deg)
                pole_count += 1
                pole_count2 = 0
                for harmonic_deg2 in 0:convert(Int, degree/2)
                    multiindex_length2 = length(get_multiindices(d, harmonic_deg2))
                    for harmonic_ord2 in 1:multiindex_length2
                        # for orth_poly_idx2 in 0:M
                        for orth_poly_idx2 in 0:(degree-harmonic_deg2)
                           pole_count2 += 1
                            if harmonic_deg != harmonic_deg2 || harmonic_ord != harmonic_ord2
                                continue
                            end
                            Cmat[pole_count, pole_count2] = integrations[(harmonic_deg, orth_poly_idx, orth_poly_idx2)]
                        end
                    end
                end
            end
        end
    end
    end
    @timeit to "eigendecompose c" begin
    evals, evecs = eigen(Cmat)
    end
    neg_bound = length(evals)
    pos_bound = 1
    for i in 1:length(evals)
        if abs(real(evals[i])) < 0.1rtol
            neg_bound = i
            break
        end
    end
    for i in length(evals):-1:1
        if abs(real(evals[i])) < 0.1rtol
            pos_bound = i
            break
        end
    end
    if pos_bound > neg_bound
        indices = vcat(collect(1:neg_bound), collect(pos_bound:length(evals)))
        evals = evals[indices]
        evecs = evecs[:, indices]
    end
    @timeit to "form output mats" begin
    new_U_mat = transpose(V_mat)[:,1:size(evecs,1)]*evecs
    new_V_mat = diagm(evals)*conj(transpose(evecs))*conj(V_mat)[1:size(evecs,1),:]
    end
    # return new_U_mat, new_V_mat
    return U_mat, V_mat
end


function degen_kern_harmonic(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
    return sandbox(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)
end



function degen_kern_nonharmonic(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, fkt_config)

end
