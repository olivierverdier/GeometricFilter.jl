using Test
using GeometricFilter
using Manifolds


import PDMats


import Random

using Infiltrator

rng = Random.default_rng()


@testset "Stochastic Motion Modes" begin
    M = Sphere(2)
    G = SpecialOrthogonal(3)
    A = RotationAction(M, G)
    x = [1., 0, 0]
    noise = IsotropicNoise(M, 1.)
    motion = ZeroMotion(M)
    smm = StochasticMotion(motion, noise, MotionMode())
    smp = StochasticMotion(motion, noise, PositionMode())
    D = ProjLogNormal(A, x, 1.)
    @test smm isa AbstractStochasticMotion{MotionMode}
    @test !(smm isa AbstractStochasticMotion{PositionMode})
    Dm = GeometricFilter.apply_noise(rng, smm, D)
    @test Dm.μ == D.μ
    Dp = GeometricFilter.apply_noise(rng, smp, D)
    @test Dp.μ != D.μ
end
