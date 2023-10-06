using Test
using GeometricFilter

using Manifolds
import Random: default_rng

rng = default_rng()

@testset "Test Multiaffine Action" begin
    dim = 3
    size = 2
    G = MultiDisplacement(dim, size)
    sel = zeros(size)
    k = rand(rng, eachindex(sel))
    sel[k] = 1.
    A = MultiAffineAction(G, sel)
    @test repr(A) == "MultiAffineAction(MultiDisplacement(3, 2), $(repr(sel)), LeftAction())"
    p = zeros(dim)
    χ = identity_element(G)
    χ.x[1][:,:] = randn(rng, dim, size)
    res = apply(A, χ, p)
    @test isapprox(res, χ.x[1][:,k])
    p_ = apply(A, Identity(G), p)
    @test isapprox(p, p_)
    ξ = zero_vector(G, Identity(G))
    computed =  apply_diff_group(A, Identity(G), ξ, p)
    expected = zeros(dim)
    @test isapprox(computed, expected)
    se = MultiDisplacement(dim,1)
    A_ = MultiAffineAction(se)
    @test isa(A_, MultiAffineAction{<:Any, typeof(base_group(se.op.action)), dim, 1})
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

