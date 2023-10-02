using PDMats

using Test
using GeometricFilter
using LinearAlgebra
using Random

rng = Random.default_rng()

@testset "Add" begin
    c = covariance_from([1; 0;;])
    @test_throws MethodError covariance_from([1,2])
    Z = zeros(2, 2)
    @test Z + c ≈ c
    @test c + Z ≈ c
    @test GeometricFilter.get_dim(c) == 2
    @test Matrix(c) == [1 0; 0 0]
    @test diag(c) == [1,0]
    complement = covariance_from([0;1;;])
    sum = c + complement
    @test sum isa Covariance
    @test sum ≈ Matrix(c) + Matrix(complement)
end

@testset "Conversion to Covariance" begin
    A = randn(2, 2)
    M = A * A'
    @test_throws MethodError Covariance(M)
    S = Symmetric(M)
    D = PDiagMat(abs2.(randn(3)))
    s = ScalMat(3, 2.0)

    @testset "Conversion $(typeof(mat))" for mat in [s, D, S]
        @test mat ≈ Covariance(mat)
    end

    @testset "Conversion from sparse diagonal" begin
        N = 20
        k = 4
        zdiag = rand(N) .+ 1
        zdiag[collect(Iterators.take(axes(zdiag, 1), k))] .= 0
        shuffle!(rng, zdiag)
        D0 = PDiagMat(sparsevec(zdiag))
        C = Covariance(D0)
        @test GeometricFilter.depth(C) == N-k
        @test C ≈ D0
        D1 = PDiagMat(zdiag)
        @test GeometricFilter.depth(Covariance(D1)) == N
    end
end

@testset "Multiplications" begin
    A = randn(2, 2)
    X = randn(2, 2)
    C = covariance_from(A)

    @test 1*C ≈ C
    @test C*1 ≈ C

    @test X*C ≈ X*Matrix(C)
    @test C*X ≈ Matrix(C)*X

    computed = X_A_Xt(C, X)
    @test computed isa Covariance
    expected = X*Matrix(C)*X'
    @test computed ≈ expected

end

