using Test
using GeometricFilter
using Manifolds

import Random
import Distributions
import PDMats

import LinearAlgebra

rng = Random.default_rng()

@testset "Multiaffine action" begin
    G = MultiDisplacement(2)
    A = MultiAffineAction(G)
    M = group_manifold(A)
    μ = zeros(manifold_dimension(M))
    dG = manifold_dimension(G)
    mat = randn(dG, dG)
    Σ = PDMats.PDMat(mat*mat')
    D = ProjLogNormal(A, μ, Σ)
    apply!(D.action, similar(D.μ), identity_element(G), D.μ)
    rand(D)
    @test_throws TypeError ProjLogNormal(switch_direction(A), μ, Σ)
end

@testset "ProjLogNormal" begin
    G = SpecialOrthogonal(3)
    M = Sphere(2)
    A = Manifolds.ColumnwiseMultiplicationAction(M, G)
    μ = [1., 0, 0]
    Σ = PDMats.ScalMat(3, 1.)
    B = DefaultOrthonormalBasis()
    dist = ProjLogNormal(A, μ, Σ, B)
    @show x = rand(rng, dist)
    @show rand(dist)
    @test scaled_distance(dist, [0., 1, 0]) == scaled_distance(dist, [0, 0, 1.])
end

@testset "action_noise" begin
    G = Orthogonal(3)
    M = Sphere(2)
    A = Manifolds.ColumnwiseMultiplicationAction(M, G)
    μ = [0, 0, 1.0]
    Σ = PDMats.PDiagMat([0.0, 0.0, 1.0])
    B = DefaultOrthonormalBasis()
    dist = ProjLogNormal(A, μ, Σ, B)
    noise = action_noise(dist)
    @test noise(rng, μ) ≈ μ
end

