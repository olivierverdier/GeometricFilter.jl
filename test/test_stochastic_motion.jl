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
    D = ProjLogNormal(A, x, 1.0)
    x0 = integrate(DataMode(), sm, x)
    xs = integrate(PositionPerturbation(rng), sm, x)
    xp = integrate(SensorPerturbation(rng), sm, x)
    @test x0 == x
    @test xs != x
    @test xp != x
    @test distance(M, xp, x) < 1e-5
    @test distance(M, xs, x) < 1e-5

    pert = GeometricFilter.sensor_perturbation(rng, sm, x)
    @test pert isa StochasticMotion
end
