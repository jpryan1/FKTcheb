# Poly will be vec of length 2p
# multiply will double check at first to make sure nothing bad happens

function multiply_polys(a::AbstractVector, b::AbstractVector)
    ans = zeros(length(a))
    # adeg = 0
    # for i in length(a):-1:1
    #     if a[i] != 0
    #         adeg=i
    #         break
    #     end
    # end
    # bdeg = 0
    # for i in length(b):-1:1
    #     if b[i] != 0
    #         bdeg=i
    #         break
    #     end
    # end
    # if adeg+bdeg > length(a)
    #     println("ERROR DROPPING TERMS") # DO NOT DELETE UNLESS TEMPORARILY FOR TIMING
    # end
    for i in 1:length(a)
        for j in 1:i
            # i-1 = j-1 + i-j
            ans[i] += a[j]*b[i-j+1]
        end
    end
    return ans
end


function integrate_poly(p::AbstractVector, b::Float64)
    tot = 0
    for i in 1:length(p)
        tot += ((b^i)/i)*p[i]
    end
    return tot
end


function poly_mat_to_mat(pm::AbstractVector{<:AbstractVector}, data_pows::AbstractVector{<:AbstractVector}, pm_lower_deg::Int64, pm_upper_deg::Int64)
    mat = zeros(length(data_pows[1]), length(pm))
    for j in 1:length(pm)
        for i in pm_lower_deg:pm_upper_deg
            mat[:,j] .+= pm[j][i]*data_pows[i]
        end
    end
    return mat
end


function poly_col_dot(poly1::AbstractVector,
                        poly2::AbstractVector,
                            data_pow_sums::AbstractVector)
    poly_dot = multiply_polys(poly1,poly2)
    tot = 0
    for i in 1:length(poly_dot)
        tot += poly_dot[i]*data_pow_sums[i]
    end
    return tot
end

function qr_poly_mat(pm::Array{Array{Float64, 1}, 1}, data_pow_sums::AbstractVector) # TODO check acc of this
    r_mat = zeros(length(pm), length(pm))
    new_polys = [copy(pm[i]) for i in 1:length(pm)]

    for i in 1:length(pm)
        current_poly = new_polys[i]
        max_norm = sqrt(max(0,poly_col_dot(current_poly,current_poly, data_pow_sums)))
        tmp = max_norm
        max_norm_idx = i
        for j in (i+1):length(pm)
            current_poly = new_polys[j]
            curr_norm = sqrt(max(0,poly_col_dot(current_poly,current_poly, data_pow_sums)))
            if curr_norm > max_norm
                max_norm = curr_norm
                max_norm_idx = j
            end
        end

        if max_norm < 1e-8
            new_polys = new_polys[1:(i-1)]
            break
        end
        cpy = copy(new_polys[i])
        new_polys[i] = copy(new_polys[max_norm_idx])
        new_polys[max_norm_idx] = cpy

        new_polys[i] = new_polys[i]/max_norm
        for j in (i+1):length(new_polys)
            dotprod = poly_col_dot(new_polys[i], new_polys[j], data_pow_sums)
            new_polys[j] .-= (dotprod) * new_polys[i]
        end
    end
    for i in 1:length(pm)
        for j in 1:length(new_polys)
            r_mat[j,i] = poly_col_dot(new_polys[j], pm[i], data_pow_sums)
        end
    end
    return new_polys, r_mat[1:length(new_polys), :]
end


function pm_mul_mat(pm::Array{Array{Float64, 1}, 1}, mat::Array{Float64, 2})
    new_polys = Array{Array{Float64,1},1}(undef, size(mat, 2))
    for j in 1:size(mat, 2)
        cur_poly = zeros(length(pm[1]))
        for i in 1:size(mat,1)
            cur_poly .+= (mat[i,j] * pm[i])
        end
        new_polys[j] = cur_poly
    end
    return new_polys
end

function pm_mul_pm(pm1::Array{Array{Float64, 1}, 1}, pm2::Array{Array{Float64, 1}, 1}, data_pow_sums::AbstractVector)
    out = zeros(length(pm1), length(pm2))
    for i in 1:size(out,1)
        for j in 1:size(out,2)
            out[i,j] = poly_col_dot(pm1[i], pm2[j], data_pow_sums)
        end
    end
    return out
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


function get_trans_table(degree, d, b, a_vals, pij)
    alpha = (d//2) - 1
    M = 2degree
    trans_table = Dict()
    for harmonic_deg in 0:convert(Int, degree/2) # TODO speed up this is so dumb
        for m in harmonic_deg:2:(degree-harmonic_deg)
            for n in harmonic_deg:2:(degree-m)
                trans_table[(harmonic_deg,m,n)] = 0
            end
        end
    end

    for harmonic_deg in 0:convert(Int, degree/2)
        gegnorm = gegenbauer_normalizer(d, harmonic_deg)
        for m in harmonic_deg:2:(degree-harmonic_deg)
            for n in harmonic_deg:2:(degree-m)
                for k3 in harmonic_deg:min(convert(Int, degree/2), min(min(n,m), degree-m))
                    for i in (n+m):2:degree
                        if mod(k3+m, 2) != 0 || mod(k3+n,2) != 0
                            continue
                        end
                        trans_table[(harmonic_deg,m,n)] += (
                                a_vals[i+1]
                                * gegnorm
                                * pij[i+1, n+m+1]
                                * (1-delta(0,i)/2)
                                * A(harmonic_deg, k3, alpha)
                                * (-2)^k3
                                * ((1.0/b)^(n+m))
                                * multinomial([convert(Int,(n-k3)/2),convert(Int,(m-k3)/2),convert(Int,k3)]...) )
                    end
                end
            end
        end
    end
    return trans_table
end

function degen_kern_harmonic(lkern, x_vecs::Array{Array{Float64,1},1}, rtol, to)
    DCT_N = 100

    centroid = sum(x_vecs)/length(x_vecs)
    for i in 1:length(x_vecs)
        x_vecs[i] .-= centroid
    end
    b = 2maximum(norm.(x_vecs))

    degree = guess_hdf_err(lkern, b,  DCT_N, rtol)
    pij    = get_pij_table(degree+1)
    d      = length(x_vecs[1])

    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b,DCT_N)
    end

    rj_hyps = cart2hyp.(x_vecs)
    trans_table = get_trans_table(degree, d, b, a_vals, pij)

    radial_mats = []
    top_sing = -1
    rank = 0
    powers, x_data_pow_sums = powers_of_norms(x_vecs, 2*(degree+d+1))
    powers_poly_rep = Array{Array{Float64,1},1}(undef, size(powers, 2))
    for i in 1:size(powers,2)
        poly = zeros(2*(degree+d+1))
        poly[i] = 1
        powers_poly_rep[i]=poly
    end
    ptp = pm_mul_pm(powers_poly_rep, powers_poly_rep, x_data_pow_sums)
    radial_y = [zeros(2*(degree+d+1)) for i in 1:(degree+1)]
    for i in 1:(degree+1)
        radial_y[i][i]=1
    end

    poly_mats = []
    diag_mats = []
    num_harmonic_orders_needed = convert(Int, degree/2)
    for harmonic_deg in 0:convert(Int, degree/2)
        polynum = convert(Int64, 1+(degree-2harmonic_deg)/2)
        x_polys = Array{Array{Float64,1},1}(undef, polynum)
        y_polys = Array{Array{Float64,1},1}(undef, polynum)

        for m in harmonic_deg:2:(degree-harmonic_deg)
            cur_poly = zeros(2*(degree+d+1))
            for n in harmonic_deg:2:(degree-m)
                cur_poly[n+1] = trans_table[(harmonic_deg,m,n)]
            end
            poly_idx = convert(Int64,1+(m-harmonic_deg)/2)
            x_polys[poly_idx] = cur_poly

            y_poly = zeros(2*(degree+d+1)) #TODO check that this high degree is needed everywhere
            y_poly[m+1] = 1
            y_polys[poly_idx] = y_poly
        end

        radial_x = x_polys
        radial_y = y_polys
        @timeit to "y qr" curr_qy, curr_ly = qr_poly_mat(radial_y, x_data_pow_sums) # IDEA move out of loop for speed

        # laq = curr_ly * pm_mul_pm(radial_x, curr_qy, x_data_pow_sums)
        pm_upper_deg1 = 0
        pm_lower_deg1 = 2degree + 2
        for i in 1:length(radial_x)
            for j in 1:length(radial_x[i])
                if radial_x[i][j] != 0
                    pm_upper_deg1 = max(pm_upper_deg1, j)
                    pm_lower_deg1 = min(pm_lower_deg1, j)
                end
            end
        end
        radx = permutedims(hcat(radial_x...))[:, pm_lower_deg1:pm_upper_deg1]

        pm_upper_deg2 = 0
        pm_lower_deg2 = 2degree + 2
        for i in 1:length(curr_qy)
            for j in 1:length(curr_qy[i])
                if curr_qy[i][j] != 0
                    pm_upper_deg2 = max(pm_upper_deg2, j)
                    pm_lower_deg2 = min(pm_lower_deg2, j)
                end
            end
        end
        rady = transpose(permutedims(hcat(curr_qy...)))[pm_lower_deg2:pm_upper_deg2,:]

        laq = (curr_ly
                *radx
                *ptp[ pm_lower_deg1:pm_upper_deg1, pm_lower_deg2:pm_upper_deg2]
                *rady)
        # laq = curr_ly*permutedims(hcat(radial_x...))*ptp*transpose(permutedims(hcat(curr_qy...)))

        smid, umid = eigen(Hermitian(laq))
        if harmonic_deg == 0
            top_sing = abs(smid[end])
        end
        leftmat = pm_mul_mat(curr_qy,umid)
        desired_indices = []
        for i in length(smid):-1:1
            if (abs(smid[i])/top_sing) >= (rtol)
                push!(desired_indices, i)
            end
        end
        smid = smid[desired_indices]
        leftmat = leftmat[desired_indices]
        if length(smid) == 0
            num_harmonic_orders_needed = harmonic_deg-1
            break
        end
        rank += length(smid)*get_num_multiindices(d, harmonic_deg)

        push!(poly_mats, leftmat)
        push!(diag_mats, smid)
    end

    U_mat = Array{Float64, 2}(undef, length(x_vecs),rank)
    diag_mat = Array{Float64, 1}(undef, rank)

    cur_idx = 0
    normalizer_table = hyper_normalizer_table(d, num_harmonic_orders_needed)
    for harmonic_deg in 0:num_harmonic_orders_needed
        rad_diag = diag_mats[harmonic_deg+1]
        harmonic_sz = get_num_multiindices(d, harmonic_deg)
        for h in 1:harmonic_sz
            for m in 1:length(rad_diag)
                cur_idx += 1
                diag_mat[cur_idx] = rad_diag[m]
            end
        end
    end

    cur_idx=0
    for harmonic_deg in 0:num_harmonic_orders_needed
        pm_mul = poly_mats[harmonic_deg+1]
        if length(pm_mul) == 0
            break
        end
        mis = get_multiindices(d, harmonic_deg)
        x_harmonics = Array{Float64,2}(undef, length(x_vecs),length(mis))
        if d > 2
            x_harmonics .= hyperspherical.(rj_hyps, harmonic_deg, permutedims(mis), Val(false))
            x_harmonics .= (x_harmonics ./ normalizer_table[harmonic_deg+1, 1:size(x_harmonics,2)]')
        elseif d == 2
            x_harmonics .= hypospherical.(rj_hyps, harmonic_deg, permutedims(mis))
        end
        pm_upper_deg = 0
        pm_lower_deg = 2degree + 2
        for i in 1:length(pm_mul)
            for j in 1:length(pm_mul[i])
                if pm_mul[i][j] != 0
                    pm_upper_deg = max(pm_upper_deg, j)
                    pm_lower_deg = min(pm_lower_deg, j)
                end
            end
        end
        pm_mul = transpose(permutedims(hcat(pm_mul...)))
        radial_x = (powers[:, pm_lower_deg:pm_upper_deg]
            *pm_mul[pm_lower_deg:pm_upper_deg,:])

        for harmonic_ord in randperm(size(x_harmonics, 2))
            for m in 0:(size(radial_x,2)-1)
                cur_idx += 1
                U_mat[:, cur_idx] .= (
                    radial_x[:, m+1]
                    .* x_harmonics[:, harmonic_ord])
            end
        end
    end
    return U_mat, diag_mat
end


function powers_of_norms(x::AbstractVector{<:AbstractVector}, max_pow::Int)
    n = length(x)
    powers = ones(eltype(x[1]), n, max_pow)
    norms = [norm(xi) for xi in x]
    norms2 = copy(norms)
    for j in 2:max_pow
        for i in 1:n
            powers[i, j] = norms2[i]
            norms2[i] *= norms[i]
        end
    end
    power_sums = vec(sum(powers, dims = 1))
    return powers, power_sums
end
