using Test
using Manifolds

"""
Helper function used for testing flat affine motions.
"""
function flat_affine_motion(
    lin, trans,
    # B::AbstractBasis
)
    dim = size(lin, 2)
    G = TranslationGroup(dim)
    V = Euclidean(dim)
    A = TranslationAction(V, G)
    f(u) = lin * u + trans
    return AffineMotions.AffineMotion(A, f, x -> lin * x)
end

@testset "Flat Motion" begin
    @testset "Scaling+Translation" begin
        dim = 2
        A = rand(rng, dim, dim)
        b = rand(rng, dim)
        m = flat_affine_motion(A, b)
        m_ = FlatAffineMotion(A, b)
        p = rand(rng, dim)
        @test m * p ≈ m_ * p
    end
    @testset "Flat Translation" begin
        dim = 2
        t = rand(rng, dim)
        m = RigidMotion(get_flat_action(dim), t)
        m_ = FlatAffineMotion(zeros(dim, dim), t)
        p = rand(rng, dim)
        @test m * p ≈ m_ * p
    end
end
