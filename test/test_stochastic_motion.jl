using Test
using GeometricFilter
using Manifolds


import PDMats


import Random

rng = Random.default_rng()


@testset "Stochastic Motion Modes" begin
    M = Sphere(2)
    G = SpecialOrthogonal(3)
    A = RotationAction(M, G)
    x = [1., 0, 0]
    noise = ActionNoise(A, 1e-15)
    motion = ZeroMotion(A)
    sm = StochasticMotion(motion, noise)
    @test_throws MethodError StochasticMotion(motion, IsotropicNoise(M, 1.0))
    D = ActionDistribution(A, x, 1.0)
    x0 = simulate(sm, x, DataMode())
    xs = simulate(sm, x, PositionPerturbation(rng))
    xp = simulate(sm, x, SensorPerturbation(rng))
    @test x0 == x
    @test xs != x
    @test xp != x
    @test distance(M, xp, x) < 1e-5
    @test distance(M, xs, x) < 1e-5

    pert = GeometricFilter.sensor_perturbation(rng, sm, x)
    @test pert isa StochasticMotion
end



@testset "Rigid Perturbation" begin
    G = SpecialOrthogonal(3)
    S = Sphere(2)
    A = RotationAction(S, G)
    x = [1.0, 0, 0]
    σ = 4.0
    BG = DefaultOrthogonalBasis()
    noise = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ), BG)
    @test GeometricFilter.rigid_perturbation(rng, noise, x) isa RigidMotion

    @testset "Degenerate Covariance $cov" for cov in [
        PDMatsSingular.Covariance(PDiagMat(SparseArrays.sparsevec([1, 0, 0]))),
        PDiagMat([1, 0, 0])
    ]
        dnoise = ActionNoise(A, cov)
        rmot = GeometricFilter.rigid_perturbation(rng, dnoise, x)
    end
end
