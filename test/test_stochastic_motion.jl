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
    noise = ActionNoise(A, 1.)
    motion = ZeroMotion(A)
    smm = StochasticMotion(motion, noise, GeometricFilter.MotionMode())
    smp = StochasticMotion(motion, noise, GeometricFilter.PositionMode())
    @test_throws MethodError StochasticMotion(motion, IsotropicNoise(M, 1.0), GeometricFilter.PositionMode())
    D = ProjLogNormal(A, x, 1.)
    @test smm isa AbstractStochasticMotion{GeometricFilter.MotionMode}
    @test !(smm isa AbstractStochasticMotion{GeometricFilter.PositionMode})
    Dm = GeometricFilter.apply_noise(rng, smm, D)
    @test Dm.μ == D.μ
    Dp = GeometricFilter.apply_noise(rng, smp, D)
    @test Dp.μ != D.μ

    pert = GeometricFilter.sensor_perturbation(rng, smm, x)
    @test pert isa StochasticMotion
end
