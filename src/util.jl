
# Struct of config params for hdfcheb approximation
struct hdf_config
    hdf_deg::Int
    d::Int
    dct_n::Int
    to::TimerOutput
    rtol::Float64
end


hdf_config(hdf_deg, d, dct_n, to) = hdf_config(hdf_deg, d, dct_n, to, 1e-6)


function guess_hdf_err(lkern,b,  dct_n, tol)
    max_deg = 70
    a_vals = zeros(max_deg) # kern's coefs in cheb poly basis
    for i in 0:(max_deg-1)
        a_vals[i+1] = dct(lkern, i, b, dct_n)
    end
    a_vals[1]/=2
    xs = collect(i/100.0 for i in 0:100)
    truths = lkern.(b*xs)
    for d in 0:(max_deg-1)
        guess_poly = ChebyshevT(a_vals[1:(d+1)])
        guesses = guess_poly.(xs)
        max_err = maximum(abs.(guesses .- truths))
        if max_err < tol
            return d
        end
    end
    println("CANNOT ACHIEVE TOL")
    return max_deg
end



# Table of coefficients for chebyshev polynomials
function get_pij_table(p_size::Int)
    pij = zeros(Int, p_size, p_size)
    pij[1,1] = 1
    if p_size==1 return pij end
    pij[2,2] = 1
    for i in 3:p_size
        for j in 1:i
            if j == 1
                pij[i,j] = -pij[i-2,j]
            else
                pij[i,j] = 2pij[i-1,j-1] - pij[i-2,j]
            end
        end
    end
    return pij
end

# Discrete cosine transform of function f defined on [-b, b]. dct_n is
# a truncation parameter.
function dct(f, k, b, dct_n)
    total = 0
    for j in 0:dct_n
        val = f(b*cos(pi*j/dct_n)) * cos(pi*k*j/dct_n)
        if j==0 || j==dct_n
                val*=0.5
        end
        total += val
    end
    return (2.0/dct_n)*total
end


function delta(x,y)
    if x==y return 1 else return 0 end
end

# This is the mathcal{T} term in the cheb expansion
function transform_coef(k3, k2, k1, degree, pij, a_vals, b, xp)
    tot3 = 0
    for i in convert(Int,2k1+2k2+2k3):degree
        tot3 +=(pij[i+1, convert(Int, 2k1+2k2+2k3+1)]
            *a_vals[i+1]
            *(1-delta(0,i)/2))
    end
    return tot3 * multinomial(xp...)*multinomial([k1,k2,k3]...)*((1.0/b)^(2k1+2k2+2k3))*((-2.0)^k3)
end



############################ miscellaneous helpers #############################
using Combinatorics: doublefactorial
doublefact(n::Int) = (n < 0) ? BigInt(1) : doublefactorial(n)
