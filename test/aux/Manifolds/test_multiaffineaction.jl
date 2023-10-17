using Test
using GeometricFilter

using Manifolds
import Random: default_rng

rng = default_rng()

inverse_if(::Any, χ, ::LeftAction) = χ
inverse_if(G, χ, ::RightAction) = inv(G, χ)

@testset "Test Multiaffine Action" begin
    dim = 3
    size = 2
    G = MultiDisplacement(dim, size)
    sel = zeros(size)
    k = rand(rng, eachindex(sel))
    sel[k] = 1.0
    p = zeros(dim)
    @testset for conv in [LeftAction(), RightAction()]
        A = MultiAffineAction(G, sel, conv)
        @test repr(A) == "MultiAffineAction(MultiDisplacement(3, 2), $(repr(sel)), $(repr(conv)))"
        χ = identity_element(G)
        χ.x[1][:, :] = randn(rng, dim, size)
        res = apply(A, χ, p)
        expected = inverse_if(G, χ, conv).x[1][:, k]
        @test isapprox(res, expected)

        p_ = apply(A, Identity(G), p)
        @test isapprox(p, p_)

        @testset "Diff" begin
            ξ = zero_vector(G, Identity(G))
            computed = apply_diff_group(A, Identity(G), ξ, p)
            expected = zeros(dim)
            @test isapprox(computed, expected)
        end
    end
    se = MultiDisplacement(dim, 1)
    A_ = MultiAffineAction(se)
    @test A_.selector == [1]
end

@testset "MultiAffineAction apply" begin
    dim = 3
    size = 2
    G = MultiDisplacement(dim, size)
    sel =randn(size)
    A = MultiAffineAction(G, sel)
    χ = rand(rng, G)
    p = rand(group_manifold(A))
    computed = apply(A, χ, p)
    M,R = submanifold_components(G,χ)
    expected = M*sel + R*p
    @test computed ≈ expected
    @test apply(switch_direction(A), χ, p) ≈ apply(A, inv(G,χ), p)
end

