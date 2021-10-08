
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
        *bigfact)
end

r = Sym("r")

function cheb_fkt(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, truth_mat, fkt_config)


    to     = fkt_config.to

    @timeit to "init" begin

    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij = get_pij_table(degree+1)
    d      = fkt_config.d
    dct_n  = fkt_config.dct_n
    num_points = length(x_vecs)
    dover2 = convert(Int, degree/2)
    pole_count= binomial(dover2+d-1, d) + binomial(dover2+d, d)
    # pole_count = binomial(convert(Int, floor(degree/2))+d+1,d+1)
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, dct_n)
    end


    trunc_param = degree

    weight = r^2
    polynomials = [Sym(1)]
    polynomials[1] /= sqrt(integrate(weight*(polynomials[1]^2), (r,0,1)))
    for i in 2:trunc_param
        current = r^(i-1)
        for j in 1:(i-1)
            current -= polynomials[j] * (integrate(weight*polynomials[j]*r^(i-1), (r, 0, 1))
                    / integrate(weight*polynomials[j]^2, (r, 0, 1)))
        end
        norm = sqrt(integrate(weight*current^2, (r,0,1)))
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
    for j in 0:dover2
        for m in j:(degree-j)
            if mod(m+j, 2) != 0
                continue
            end
            for n in j:(degree-m)
                if mod(j+n, 2) != 0
                    continue
                end
                for k3 in j:min(m,n)
                    if mod(k3+n, 2) != 0
                        continue
                    end
                    for i in (n+m):degree
                        m1 = convert(Int,((n-k3)/2))
                        m2 = convert(Int,((m-k3)/2))
                    transtable[j+1, m+1, n+1] += (pij[i+1, n+m+1] # here
                                        * a_vals[i+1]
                                        * A(j, k3, 1//2)
                                        *(1-delta(0,i)/2) # here
                                        *multinomial([m1,m2,k3]...) #here
                                        *((1.0/b)^(n+m)) #here
                                        *((-2.0)^k3)) #here
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

    normalizer_table = squared_hyper_normalizer_table(d, dover2)


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
        elseif d == 2
            hyp_harms_k_a .= hypospherical.(ra_hyps, j, permutedims(multiindices)) # needs to be normalized
        end

        for idx in 1
            if abs(1-dot(hyp_harms_k[idx, :], conj(hyp_harms_k_a[idx,:]))) > 1e-5
                println(dot(hyp_harms_k[idx, :], conj(hyp_harms_k_a[idx,:])))
            end
        end

        for h in 1:size(hyp_harms_k, 2)
            for m in j:2:(degree-j)
                pole_counter +=1
                for n in j:2:(degree-m)
                    for (x_idx, x_vec) in enumerate(x_vecs)
                        U_mat[x_idx, pole_counter] += ((norm(x_vec)^(n))
                                *transtable[j+1, m+1, n+1]
                                * hyp_harms_k[x_idx, h])
                    end
                end
                for (y_idx, y_vec) in enumerate(y_vecs)
                    V_mat[pole_counter, y_idx] = ((norm(y_vec)^(m))* hyp_harms_k_a[y_idx, h])
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
