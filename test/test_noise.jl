using Manifolds: LinearAlgebra
using Test
using GeometricFilter
using Manifolds

import PDMats
import Random

function Manifolds.apply_diff_group(
    ::RotationAction{N,F,LeftAction},
    ::Identity,
    X,
    p,
    ) where {N,F}
    return X * p
end

@testset "Constant Function" begin
    value = 2.0
    f = GeometricFilter.ConstantFunction(value)
    @test f(0) == value
    @test (2*f)(0) == 2*value
end


@testset "Test Noise" begin
    G = Orthogonal(3)
    A = GroupOperationAction(G)
    B = DefaultOrthonormalBasis()
    σ = 4.0
    N = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ),  B)
    N_ = ActionNoise(A, σ)
    N__ = sqrt(σ)*ActionNoise(A)
    @test N_ == N
    @test N__ == N
    # get_basis_at(N, Identity(G))
    @show get_covariance_at(N, Identity(G))
    sn = .5*N
    @show sc = get_covariance_at(sn, Identity(G), B)
end

@testset "ActionNoise Basis" begin
    G = SpecialOrthogonal(3)
    M = Sphere(2)
    A = RotationAction(M, G)
    # x = rand(M)
    x = [1., 0, 0]
    noise = ActionNoise(A, 1.)
    B = get_basis_at(noise, x)
    @show get_vector(M, x, [1., 0], B)
    @show B
end

@testset "NoNoise" begin
    dim = 2
    M = Sphere(dim)
    n = NoNoise(M)
    x = rand(M)
    rng = Random.default_rng()
    @test x == n(rng, x)
    @test n == 2.0*n
    @test get_covariance_at(n, x) == zeros(dim,dim)
end

@testset "Flat Action Noise" begin
    lin = [0 1.0; 0 0]
    trans = zeros(2)
    motion = FlatAffineMotion(lin, trans)
    A = get_action(motion)

    x0 = [0.0, 20]
    observer = LinearObserver([1.0 0])
    onoise = ActionNoise(TranslationAction(Euclidean(1), TranslationGroup(1)), 10.0)
    @test get_covariance_at(onoise, 0, DefaultOrthonormalBasis()) isa PDMats.AbstractPDMat
end

@testset "Rescale" begin
    G = Orthogonal(3)
    A = GroupOperationAction(G)
    B = DefaultOrthonormalBasis()
    in = IsotropicNoise(G, x->1.)
    in_ = IsotropicNoise(G, 1.0)
    @test get_covariance_at(2*in, Identity(G), B) == 4*LinearAlgebra.diagm(ones(manifold_dimension(G)))
    @test_throws MethodError get_covariance_at(in, Identity(G), DefaultOrthogonalBasis())
    an = ActionNoise(A, PDMats.ScalMat(3, 1.), B)
    an_ = ActionNoise(A, PDMats.ScalMat(3, 1.), B)
    @test get_covariance_at(2*an, Identity(G), B) == 4*LinearAlgebra.diagm(ones(manifold_dimension(G)))
end

@testset "Test Action Noise" begin
    G = SpecialOrthogonal(3)
    S = Sphere(2)
    A = RotationAction(S, G)
    x = [1., 0, 0]
    BG = DefaultOrthogonalBasis()
    BM = DefaultOrthonormalBasis()
    P = GeometricFilter.get_proj_matrix(A, x, BG, BM)
    @show P
    # noise = ActionNoise(A, x->Matrix{Float64}(LinearAlgebra.I, 3, 3), BG)
    σ = 4.0
    noise = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ), BG)
    cov = get_covariance_at(noise, x, BM)
    @test GeometricFilter.get_lie_covariance_at(noise, x, BG) == PDMats.ScalMat(3, σ)
    @test cov == PDMats.ScalMat(2, σ)
end
