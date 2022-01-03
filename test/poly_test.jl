module FactorTest

using Test
using FKTcheb
using SymPy


@testset "eval" begin
    r = Sym("r")
    p = 3.4 + 9.2r - 1.6r^2
    a = [3.4 9.2 -1.6]
    x = rand()
    @test isapprox(subs(p, r=>x), evaluate_poly(a, x), rtol = 1e-9)
end


@testset "mul" begin
    r = Sym("r")
    a1 = rand(8)
    a2 = rand(8)
    a1[5:end] .= 0
    a2[5:end] .= 0

    p1 = a1[1] + a1[2]*r+a1[3]*r^2 + a1[4]*r^3
    p2 = a2[1] + a2[2]*r+a2[3]*r^2 + a2[4]*r^3

    a3 = multiply_polys(a1, a2)
    p3 = p1*p2

    x = rand()
    @test isapprox(subs(p3, r=>x), evaluate_poly(a3, x), rtol = 1e-9)
end

end
