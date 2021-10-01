function cauchy_fkt(lkern, x_vecs::Array{Array{Float64,1},1}, y_vecs::Array{Array{Float64,1},1}, truth_mat, fkt_config)


    to     = fkt_config.to
    @timeit to "init" begin
    b      = fkt_config.b
    degree = fkt_config.fkt_deg
    pij = get_pij_table(degree+1)

    transtable = zeros(Float64, degree+1, degree+1, degree+1, degree+1)
    for k3 in 0:convert(Int,(degree/2))
        for k2 in 0:convert(Int, (floor(degree/2)-k3))
            for i in convert(Int,2k2+2k3):degree
                for k1 in 0:convert(Int, floor(i/2)-k2-k3)
                    transtable[i+1, k1+1, k2+1, k3+1] = (pij[i+1, convert(Int, 2k1+2k2+2k3+1)] # here
                                        *(1-delta(0,i)/2) # here
                                        *multinomial([k1,k2,k3]...) #here
                                        *((1.0/b)^(2k1+2k2+2k3)) #here
                                        *((-2.0)^k3)) #here
                end
            end
        end
    end

    d      = fkt_config.d
    dct_n  = fkt_config.dct_n
    num_points = length(x_vecs)
    pole_count = binomial(convert(Int, floor(degree/2))+d+1,d+1)
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, dct_n)
    end

    U_mat = zeros(Float64, num_points,pole_count)
    V_mat = zeros(Float64, pole_count,num_points)
    pole_counter=0
    end
    for k3 in 0:convert(Int,(degree/2))
        for xp in multiexponents(d,convert(Int,k3))
            mn = multinomial(xp...)
            for k2 in 0:convert(Int, (floor(degree/2)-k3))
                pole_counter +=1
                @timeit to "x loop" begin
                    for (x_idx, x_vec) in enumerate(x_vecs)
                    # for x_idx in 1:length(x_vecs)
                        xxp = prod(x_vec.^(xp))
                        for i in convert(Int,2k2+2k3):degree
                            for k1 in 0:convert(Int, floor(i/2)-k2-k3)
                                U_mat[x_idx, pole_counter] += (a_vals[i+1]*
                                        (norm(x_vec)^(2k1))
                                        *transtable[i+1, k1+1, k2+1, k3+1])
                            end
                        end
                        U_mat[x_idx, pole_counter] *= mn*xxp
                    end
                end
                @timeit to "y loop" begin
                    for (y_idx, y_vec) in enumerate(y_vecs)
                    # for y_idx in 1:length(y_vecs)
                        V_mat[pole_counter, y_idx] = ((norm(y_vec)^(2k2))*prod(y_vec.^(xp)))
                    end
                end
            end
        end
    end
    return U_mat,  V_mat
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
