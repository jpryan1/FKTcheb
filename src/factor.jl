# Poly will be vec of length 2p
# multiply will double check at first to make sure nothing bad happens

function falling(x,r)
    if r==0 return 1 end
    prod = 1
    for i in 0:(r-1)
        prod *= (x+i)
    end
    return prod
end


function Ar(r)
    if r==0 return 1 end
    prod = 1
    for i in r:-1:1
        prod *= (0.5+r-i)/i
    end
    return prod
end

# three things depend on polynomial
# shifted leg table
# multiply polys
# legpol call


function heart(p,q,r)
    return ((Ar(r)*Ar(p-r)*Ar(q-r)*(2p+2q-4r+1))/(Ar(p+q-r)*(2p+2q-2r+1)))
end


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

    for i in 1:adeg
        for j in 1:bdeg
            if a[i]*b[j] != 0
                top = max(i,j)
                bot = min(i,j)
                p=top-1
                q=bot-1
                for r in 0:q
    # on the product of two legendre polynomials
    ans[p+q-2r+1] += a[i]*b[j]*heart(p,q,r) * sqrt(2p+1)*sqrt(2q+1)/sqrt(2(p+q-2r)+1)
    # ans[p+q-2r+1] += a[i]*b[j]*heart(p,q,r)
    #NORMALIZATION
                end
            end
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


function normalized_leg(legpoly,x)
    tot = 0
    for i in 1:length(legpoly)
        leg = zeros(i)
        leg[i] = 1
        legpol =Legendre(leg)
        tot += legpol(x)*sqrt(2(i-1)+1)*legpoly[i]
    end
    return tot
end


function poly_qr(polys, x_vec_norms, data_pows, degree)
    k_to_data_pow_sums = Dict()
    k_to_data_pows = Dict()
    for k in 1:length(polys)
        x_vec_norms_k = x_vec_norms[k:end]
        data_pows_k = Array{Array{Float64,1}, 1}()
        for i in 1:(3degree)
            leg = zeros(i)
            leg[i] = 1
            legpol = Legendre(leg)

            push!(data_pows_k, sqrt(2(i-1)+1)*legpol.(2x_vec_norms_k .- 1))
            # push!(data_pows_k, legpol.(2x_vec_norms_k .- 1))
            #NORMALIZATION
        end
        data_pow_sums_k = [sum(data_pows_k[i]) for i in 1:length(data_pows_k)]
        k_to_data_pow_sums[k] = data_pow_sums_k
    end
    # R = zeros(length(polys), length(polys))
    R = zeros(length(x_vec_norms), length(polys))
    vs = []
    for k in 1:(length(polys))
        x_poly = polys[k]
        # println("x poly ", x_poly)
        # println(" dot product norm: ",poly_col_dot(x_poly,x_poly,k_to_data_pow_sums[k]))


#TODO DELETE COMPLEX
        xn = sqrt(Complex(poly_col_dot(x_poly,x_poly,k_to_data_pow_sums[k])))
        # println("Guess norm ", xn)
        vk = poly_mat_to_mat([x_poly],  data_pows, 1, length(data_pows))[k:end, 1]
        # println("True norm ", norm(vk))
        # xn=norm(vk)

        # x1 = Legendre(x_poly)(2x_vec_norms[k]-1)
        x1 = normalized_leg(x_poly,2x_vec_norms[k][1]-1)
        #NORMALIZATION

        sn = sign(x1)
        alpha = sn*xn
        vkn = sqrt(xn^2 + alpha^2 + 2alpha*x1)


        vk[1] += alpha
        push!(vs, vk/vkn)

        tvec = zeros(length(polys)-k)
        for i in (k+1):length(polys)
            truth = dot(poly_mat_to_mat([x_poly],  data_pows, 1, length(data_pows))[k:end, 1],
                        poly_mat_to_mat([polys[i]],  data_pows, 1, length(data_pows))[k:end, 1])

            tvec[i-k] = poly_col_dot(x_poly, polys[i], k_to_data_pow_sums[k])
            # tvec[i-k]=truth
            # println("col dot guess ", tvec[i-k], " truth ", truth)
        end

        #NORMALIZATION
        # row_vec = [Legendre(p)(2x_vec_norms[k][1]-1) for p in polys[(k+1):end]]
        row_vec = [normalized_leg(p,2x_vec_norms[k][1]-1) for p in polys[(k+1):end]]


        right_mat = transpose(tvec)/vkn^2 + ((alpha*transpose(row_vec))/(vkn^2))

        R[k,k] = -sn*xn
        R[k, (k+1):end] .= row_vec .- transpose(2*(x1+alpha)*right_mat)

        for j in (k+1):length(polys)
            polys[j] .-= 2*right_mat[1, j-k]*polys[k]
        end
    end
    return vs, R
end


function q_poly_mul(vs, B)
    C = copy(B)
    for k in (length(vs)):-1:1
        C[k:end, :] -= 2*(vs[k]*(transpose(vs[k])*C[k:end,:]))
    end
    return C
end

function qtrans_poly_mul(vs, B)
    C = copy(B)
    for k in 1:length(vs)
        C[k:end, :] -= 2*(vs[k]*(transpose(vs[k])*C[k:end,:]))
    end
    return C
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
    # println("pair ",poly1, " ", poly2, "  turns to")
    # println("Poly dot ",poly_dot)
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

    @timeit to "centering1" centroid = sum(x_vecs)/length(x_vecs)

    @timeit to "centering2" begin
    for i in 1:length(x_vecs)
        x_vecs[i] .-= centroid
    end
end

    # for i in 1:length(x_vecs)
    #     x_vecs[i] -= centroid
    # end
    @timeit to "centering3" b = 2maximum(norm.(x_vecs))

    @timeit to "guess fkt error " degree = guess_fkt_err(lkern, b,  DCT_N, rtol)
    pij    = get_pij_table(degree+1)
    # lij    = get_leg_table(degree+1)
    lij    = get_shifted_leg_table(degree+1)
    # println(lij)
    d      = length(x_vecs[1])
    a_vals = zeros(degree+1) # kern's coefs in cheb poly basis
    for i in 0:(degree)
        a_vals[i+1] = dct(lkern, i, b,DCT_N)
    end

    M = 2degree

    @timeit to "cart2hyp" rj_hyps = cart2hyp.(x_vecs)
    @timeit to "trans table" trans_table = get_trans_table(degree, d, b, a_vals, pij)


    radial_mats = []
    top_sing = -1
    rank = 0
    x_vec_norms = norm.(x_vecs)
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
    diag_mats = []
    num_harmonic_orders_needed = convert(Int, degree/2)
    lijinv = inv(lij)
    for harmonic_deg in 0:convert(Int, degree/2)
        # polynum = convert(Int64, 1+(degree-2harmonic_deg)/2)
        polynum = degree-harmonic_deg+1

        x_polys = Array{Array{Float64,1},1}(undef, polynum)
        y_polys = Array{Array{Float64,1},1}(undef, polynum)
        my_y_polys = Array{Array{Float64,1},1}(undef, polynum)

        # for m in harmonic_deg:2:(degree-harmonic_deg)
        poly_idx = 0
        for m in 0:(degree-harmonic_deg)
            cur_poly = zeros(2*(degree+d+1))
            # for n in harmonic_deg:2:(degree-m)
            for n in harmonic_deg:2:(degree-max(m,harmonic_deg))
                for i in max(m,harmonic_deg):(degree-n)
                    if mod(i+n,2)==1 continue end
                    # cur_poly[n+1] += A(m,i,1//2)*trans_table[(harmonic_deg,i,n)]
                    cur_poly[n+1] += lijinv[i+1,m+1]*trans_table[(harmonic_deg,i,n)]
                end
            end
            # poly_idx = convert(Int64,1+(m-harmonic_deg)/2)
            poly_idx += 1
            x_polys[poly_idx] = cur_poly
            #
            y_poly = zeros(2*(degree+d+1)) #TODO check that this high degree is needed everywhere
            #
            for i in 0:(degree-harmonic_deg)
                y_poly[i+1] = lij[m+1,i+1]
            end
            # println(y_poly)
            # y_poly[m+1] = 1
            y_polys[poly_idx] = y_poly

            my_y_poly = zeros(2*(degree+d+1)) #TODO check that this high degree is needed everywhere

            # for i in 0:(degree-harmonic_deg)
            #     my_y_poly[i+1] = lij[m+1,i+1]
            # end
            # println(y_poly)
            my_y_poly[m+1] = 1
            my_y_polys[poly_idx] = my_y_poly
        end

        radial_x = x_polys
        rpx = poly_mat_to_mat(radial_x, data_pows, 1, length(data_pows))

        # println("COND ",cond(rpy))
        # println("COND ",cond(transpose(rpy)*rpy))
        my_data_pows = Array{Array{Float64,1}, 1}()
        for i in 1:(2degree+2)
            leg = zeros(i)
            leg[i] = 1
            legpol = Legendre(leg)
            push!(my_data_pows, sqrt(2(i-1)+1)*legpol.(2x_vec_norms.-1))
            # push!(my_data_pows, legpol.(2x_vec_norms.-1))
            #NORMALIZATION
        end

        # rpy = poly_mat_to_mat(my_y_polys,  my_data_pows, 1, length(my_data_pows))


        curr_qy, curr_ly = poly_qr(my_y_polys, x_vec_norms, my_data_pows, degree)

        # @timeit to "y qr" curr_qy, curr_ly = qr_poly_mat(radial_y, x_data_pow_sums) # IDEA move out of loop for speed
        # curr_qy, curr_ly, p = qr(rpy, Val(true))

        # curr_qy = curr_qy[:, 1:qrrank]
        # println([curr_ly[i,i] for i in 1:length(p)])
        # curr_ly  = curr_ly[:,invperm(p)]
        laq = curr_ly *transpose(qtrans_poly_mul(curr_qy, rpx))
        # laq = curr_ly * pm_mul_pm(radial_x, curr_qy, x_data_pow_sums)
        # laq = Array(curr_ly)*transpose(rpx)*Array(curr_qy)

        println("laq ",norm(laq-transpose(laq)))
        smid, umid = eigen(Hermitian(laq))

        if harmonic_deg == 0
            top_sing = abs(smid[end])
        end
        #
        leftmat = zeros(length(x_vec_norms), size(umid,2))
        leftmat[1:size(umid,1), 1:size(umid,2)] .= umid
        leftmat = q_poly_mul(curr_qy, leftmat)
        # leftmat = pm_mul_mat(curr_qy, umid)
        # leftmat = curr_qy*umid

        desired_indices = []
        for i in length(smid):-1:1
            if (abs(smid[i]) / top_sing) >= rtol
                push!(desired_indices, i)
            end
        end
        smid = smid[desired_indices]


        # leftmat = leftmat[desired_indices]
        leftmat = leftmat[:,desired_indices]

        if length(smid) == 0
            num_harmonic_orders_needed = harmonic_deg-1
            break
        end
        rank += length(smid)*get_num_multiindices(d, harmonic_deg)
        push!(poly_mats, leftmat)
        push!(diag_mats, smid)
    end
    end
    @timeit to "u alloc" U_mat = Array{Float64, 2}(undef, length(x_vecs),rank)
    diag_mat = Array{Float64, 1}(undef, rank)
    cur_idx = 0
    @timeit to "normalizer_table" normalizer_table = hyper_normalizer_table(d, num_harmonic_orders_needed)
    println(num_harmonic_orders_needed, " orders needed")
    @timeit to "Second loop" begin
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
    end
    cur_idx=0
    @timeit to "Third loop" begin
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

        pm_upper_deg = 0
        pm_lower_deg = 2degree
        for i in 1:length(pm_mul)
            for j in 1:length(pm_mul[i])
                if pm_mul[i][j] != 0
                    pm_upper_deg = max(pm_upper_deg, j)
                    pm_lower_deg = min(pm_lower_deg, j)
                end
            end
        end

        # pm_mul = transpose(permutedims(hcat(pm_mul...)))
        # @timeit to "pm2m rx" radial_x = (
        #      powers[:, pm_lower_deg:pm_upper_deg]
        #     *pm_mul[pm_lower_deg:pm_upper_deg,:])
        radial_x=pm_mul

        # println(norm(tmp-radial_x))
        @timeit to "populate umat " begin
        for harmonic_ord in 1:size(x_harmonics, 2)
            for m in 0:(size(radial_x,2)-1)
                cur_idx += 1
                    U_mat[:, cur_idx] .= (
                        radial_x[:, m+1]
                        .* x_harmonics[:, harmonic_ord])
                # U_mat[:, cur_idx] .= radial_x[:, m+1] .* x_harmonics[:, harmonic_ord]
            end
            end
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
    # println(powers[1:5, 2], " is the pow")
    power_sums = vec(sum(powers, dims = 1))
    data_pows = [col for col in eachcol(powers)]

    return powers, data_pows, power_sums
end
