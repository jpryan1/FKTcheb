# Poly will be vec of length 2p
# multiply will double check at first to make sure nothing bad happens

function multiply_polys(a::AbstractVector, b::AbstractVector)
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


function integrate_poly(p::AbstractVector, b::Float64)
    tot = 0
    for i in 1:length(p)
        tot += ((b^i)/i)*p[i]
    end
    return tot
end


function poly_mat_to_mat(pm::AbstractVector{<:AbstractVector}, data_pows::AbstractVector{<:AbstractVector}, pm_deg::Int64)
    mat = zeros(length(data_pows[1]), length(pm))

    for j in 1:length(pm)
        for i in 1:pm_deg
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

        if max_norm < 1e-6
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
                for i in (n+m):2:degree
                    for k3 in harmonic_deg:min(convert(Int, degree/2), min(min(n,m), degree-m))
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

function degen_kern_harmonic(lkern, x_vecs::Array{Array{Float64,1},1}, fkt_config)
    to     = fkt_config.to
    @timeit to "centering" begin
    centroid = sum(x_vecs)/length(x_vecs)
    for i in 1:length(x_vecs)
        x_vecs[i] -= centroid
    end
    b = 2maximum(norm.(x_vecs))
    end
    degree = fkt_config.fkt_deg
    pij    = get_pij_table(degree+1)
    d      = fkt_config.d
    rtol      = fkt_config.rtol
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b, fkt_config.dct_n)
    end

    M = 2degree

    @timeit to "cart2hyp" rj_hyps = cart2hyp.(x_vecs)
    @timeit to "trans table" trans_table = get_trans_table(degree, d, b, a_vals, pij)


    radial_mats = []
    top_sing = -1
    rank = 0
    @timeit to "powers_of_norms" powers, data_pows, x_data_pow_sums = powers_of_norms(x_vecs, 2*(degree+d+1))

    radial_y = [zeros(2*(degree+d+1)) for i in 1:(degree+1)]
    for i in 1:(degree+1)
        radial_y[i][i]=1
    end

    # @timeit to "y qr" qy, ly = qr_poly_mat(radial_y, x_data_pow_sums)
    # guess=poly_mat_to_mat(pm_mul_mat(qy,  ly), data_pows, length(data_pows))
    # if norm(guess-poly_mat_to_mat(radial_y, data_pows, length(data_pows)),2)/norm(poly_mat_to_mat(radial_y, data_pows, length(data_pows)),2) > 1e-5
    #     println("\n\nBIG QR ERR\n\n")
    # end
    @timeit to "stage 1" begin
    poly_mats = []
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

        laq = curr_ly * pm_mul_pm(radial_x, curr_qy, x_data_pow_sums)
        umid, smid, vmid = svd(laq)
        if harmonic_deg == 0
            top_sing = smid[1]
        end
        leftmat = pm_mul_mat(curr_qy,umid)
        for i in 1:length(smid)
            if (smid[i] / top_sing) < rtol
                smid = smid[1:(i-1)]
                leftmat = leftmat[1:(i-1)]
                break
            end
        end

        if length(smid) == 0
            num_harmonic_orders_needed = harmonic_deg-1
            break
        end
        rank += length(smid)*get_num_multiindices(d, harmonic_deg)
        pm_mul = pm_mul_mat(leftmat, diagm(sqrt.(smid)))
        push!(poly_mats, pm_mul)
    end
    end
    @timeit to "u alloc" U_mat = Array{Float64, 2}(undef, length(x_vecs),rank)

    cur_idx = 0
    @timeit to "normalizer_table" normalizer_table = hyper_normalizer_table(d, num_harmonic_orders_needed)
    println(num_harmonic_orders_needed, " orders needed")
    @timeit to "Second loop" begin
    for harmonic_deg in 0:num_harmonic_orders_needed
        pm_mul = poly_mats[harmonic_deg+1]
        if length(pm_mul) == 0
            break
        end

        mis = get_multiindices(d, harmonic_deg)
        @timeit to "get harmonics" begin
        x_harmonics = Array{Float64,2}(undef, length(x_vecs),length(mis))
        if d > 2
            x_harmonics .= hyperspherical.(rj_hyps, harmonic_deg, permutedims(mis), Val(false))
            x_harmonics .= (x_harmonics ./ normalizer_table[harmonic_deg+1, 1:size(x_harmonics,2)]')
        elseif d == 2
            x_harmonics .= hypospherical.(rj_hyps, harmonic_deg, permutedims(mis))
        end
        end

        pm_deg = 0
        for i in 1:length(pm_mul)
            for j in 1:length(pm_mul[i])
                if pm_mul[i][j] != 0
                    pm_deg = max(pm_deg, j)
                end
            end
        end
        # println("Cost ", pm_deg*length(pm_mul))
        @timeit to "pm2m rx" radial_x = poly_mat_to_mat(pm_mul, data_pows, pm_deg)
        @timeit to "populate umat " begin
        for harmonic_ord in 1:size(x_harmonics, 2)
            for m in 0:(size(radial_x,2)-1)
                cur_idx += 1
                for (x_idx, x_vec) in enumerate(x_vecs)
                    U_mat[x_idx, cur_idx] = (
                        radial_x[x_idx, m+1]
                        * x_harmonics[x_idx, harmonic_ord])
                end
            end
            end
        end
    end
    end
    return U_mat
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
    # println(powers[1:5, 2], " is the pow")
    power_sums = vec(sum(powers, dims = 1))
    powers_as_vec = [col for col in eachcol(powers)]
    return powers, powers_as_vec, power_sums
end
