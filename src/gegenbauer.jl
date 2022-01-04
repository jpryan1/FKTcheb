
# iterative Gegenbauer implementation
function gegenbauer(α::Real, n::Int, x::Real)
    α, x = promote(α, x)
    gegenbauer(α, n, x)
end
# for full performance, important to pass α as a float too
@inline function gegenbauer(α::T, n::Int, x::T) where {T <: Real}
    C1 = one(T)
    n == 0 && return C1
    C2 = 2α*x
    for k in 2:n
        @fastmath C1, C2 = C2, (2*x*(k+α-1) * C2 - (k+2α-2) * C1) / k
    end
    return C2
end

# for full performance, important to pass α as a float too
@inline function gegenbauer(α::T, n::Int, x::AbstractVector{T}) where {T <: Real}
    C1 = ones(eltype(x), length(x)) # allocates IDEA: pre-allocate C1, and C2 in hyperspherical
    n == 0 && return C1
    C2 = zero(x) # allocates
    gegenbauer!(C1, C2, α, n, x)
end

@inline function gegenbauer!(C1, C2, α::T, n::Int, x::AbstractVector{T}) where {T <: Real}
    @. C1 = 1
    @. C2 = 2α*x
    for k in 2:n
        @inbounds @simd for i in 1:length(x)
            @fastmath C1[i], C2[i] = C2[i], (2*x[i]*(k+α-1) * C2[i] - (k+2α-2) * C1[i]) / k
        end
    end
    return C2
end

function gegenbauer_normalizer(d::Int, n::Int)
    N_k_alpha = 1
    if d > 2
        N_k_alpha = inv((d+2n-2) * doublefact(d-4))
        N_k_alpha = convert(Float64, N_k_alpha)
        N_k_alpha *= iseven(d) ? (2π)^(d/2) : 2*(2π)^((d-1)/2)
    end
    return N_k_alpha
end

# iterative Chebyshev of the first or second kind
@inline function chebyshev(n::Int, x::Real, kind::Val{T} = Val(1)) where {T}
    C1 = one(x)
    n == 0 && return C1
    C2 = T*x
    for k in 2:n
        @fastmath C1, C2 = C2, (2x*C2 - C1)
    end
    return C2
end
