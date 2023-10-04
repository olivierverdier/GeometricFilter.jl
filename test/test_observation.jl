using Test
using GeometricFilter
using SparseArrays
using Random
using Manifolds

rng = Random.default_rng()

@testset "Observations" begin
    M = Sphere(2)
    T = 10
    G = SpecialOrthogonal(3)
    A = RotationAction(M, G)
    motions = fill(ZeroMotion(A), T)
    x0 = [1., 0, 0]
    states = generate_signal(motions, x0)
    observers = fill(IdentityObserver(M), T)
    onoise = IsotropicNoise(M, 1.)
    noises = fill(onoise, T)


    observations = simulate_observations(rng, states, observers, noises) do i
        return i==5 || i == 7
    end
    @test length(findnz(observations)) == 2
end

@test collect([Observation()] |> SparseVector) == [Observation()]
