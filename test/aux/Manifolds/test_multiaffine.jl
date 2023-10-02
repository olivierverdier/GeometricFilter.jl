using Test
using GeometricFilter

import ManifoldsBase as MB
using Manifolds
import Random: rand!, AbstractRNG

using Random

# TODO: this may be useful later in Motion.jl instead for some special motion cases
get_adjoint_matrix(G, vel, B::AbstractBasis) = GeometricFilter.matrix_from_lin_endomorphism(G, ξ -> lie_bracket(G, vel, ξ), B)


function my_test_group(rng, G, n=3)
    pts = [rand(rng, G) for i in 1:n]
    vels = [rand(rng, G; vector_at=Identity(G)) for i in 1:n]
    # Manifolds.test_group(G, pts, [], vels,
    Manifolds.test_group(G, pts, vels, vels,
                         test_exp_lie_log=false,
                         test_lie_bracket=true,
                         test_adjoint_action=true,
                         test_diff=false,
                         )
end


function rand_lie(rng::AbstractRNG, G)
    return rand(rng, G; vector_at=Identity(G))
end

function randn_vec(rng::AbstractRNG, G::Manifolds.GeneralUnitaryMultiplicationGroup{n,𝔽,S};
                   σ::Real=1.,
                   ) where {n,𝔽,S}
    NT = MB.allocate_result_type(G, typeof(randn_vec), ())
    M = σ*randn(rng, NT, n, n)
    ξ = M - M' # some kind of projection like the one in SkewHermitianMatrices?
    @assert is_vector(G, identity_element(G), ξ)
    return ξ
end

function randn_pt(rng::AbstractRNG, G::Manifolds.GeneralUnitaryMultiplicationGroup{n,𝔽,S};
                  σ::Real=1.,
                  ) where {n,𝔽,S}
    ξ = randn_vec(rng, G; σ)
    return exp_lie(G, ξ)
end

function rand!(rng::AbstractRNG,
               M::Manifolds.GeneralUnitaryMatrices{n,𝔽,S},
               tmp;
               vector_at=nothing,
               # σ::Real=one(eltype(tmp))
               σ=1.
               ) where {n,𝔽,S}
    G = Manifolds.GeneralUnitaryMultiplicationGroup(M)
    if vector_at === nothing
        res = randn_pt(rng, G; σ)
    elseif vector_at isa Identity
        res = randn_vec(rng, G; σ)
    else
        throw(ErrorException("Only tangent vector at identity supported"))
    end
    copyto!(tmp, res)
    return tmp
end


"""
Compute both
Ad_{exp(ξ)}ξ'
 and
exp(ad_ξ) ξ'
"""
function _compute_both(G, vel, tvel)
    χ = exp_lie(G, vel)
    right = adjoint_action(G, χ, tvel)

    lie_bracket(G, vel, tvel)
    B = DefaultOrthogonalBasis()
    der = get_adjoint_matrix(G, vel, B)
    mor = exp(der)
    tvel_coord = get_coordinates_lie(G, tvel, B)
    left = get_vector_lie(G, mor * tvel_coord, B)
    return left, right
end

function test_exp_ad(rng, G)
  @testset "exp (ad_ξ) = Ad_exp(ξ)" begin
      vel = rand_lie(rng, G)
      tvel = rand_lie(rng, G)
      left, right = _compute_both(G, vel, tvel)
      # @test isapprox(G, Identity(G), left, right)
      @test isapprox(G, identity_element(G), left, right)
  end
end


@testset "Test types" begin
    @testset "MultiDisplacement(x,y) creates proper type" begin
        dim = 4
        size = 2
        G = MultiDisplacement(dim,size)
        GM = MultiDisplacement(dim,size)
        @test isa(GM, MultiDisplacement{dim,size})
        @test !isa(GM, MultiDisplacement{dim,size+5})
        @test !isa(GM, MultiDisplacement{dim+1,size})
    end
    @testset "MultiAffine(G, size) creates proper type" begin
        GA = MultiAffine(Orthogonal(4), 3)
        @test isa(GA, MultiAffine{Orthogonal{4}, 4, 3, ℝ})
        @test !isa(GA, MultiAffine{Orthogonal{5}, 5, 3, ℝ})
        @test isa(MultiAffine(Unitary(4), 5), MultiAffine)
    end
end

function test_multi_affine(rng, G::MultiAffine{TH,dim,size,𝔽}
                           ) where {TH,dim,size,𝔽}
  @testset "Test $(repr(G))" begin
      vel = rand_lie(rng, G)
      pt = rand(rng, G)
      x = exp_lie(G, vel)
      @test is_point(G, x)
      v_ = adjoint_action(G, pt, vel)
      @test is_vector(G, identity_element(G), v_)
      GeometricFilter.affine_matrix(G, Identity(G))
      @testset "zero_element" begin
          z = zero_vector(G, Identity(G))
          z_ = zero_vector(G, identity_element(G))
          @test isapprox(G, z, z_)
      end
      @testset "from/to" begin
          ts = randn(rng, dim, size)
          χ1 = from_normal_grp(G, eachcol(ts)...)
          χ2 = from_normal_grp(G, ts)
          @test isapprox(G, χ1, χ2)
          ξ1 = from_normal_alg(G, eachcol(ts)...)
          ξ2 = from_normal_alg(G, ts)
          @test isapprox(G, Identity(G), ξ1, ξ2)
      end
      @testset "Lie Bracket & matrix" begin
          v1, v2 = [rand_lie(rng, G) for i in 1:2]
          m1, m2 = [screw_matrix(G, v) for v in [v1,v2]]
          comm = m1*m2 - m2*m1
          expected = ArrayPartition(submanifold_components(G, comm)...)
          computed = lie_bracket(G, v1, v2)
          @test isapprox(G, Identity(G),  expected, computed)
      end
      @testset "Composition & matrix" begin
          p1, p2 = [rand(rng, G) for i in 1:2]
          m1, m2 = [affine_matrix(G, p) for p in [p1,p2]]
          prod = m1*m2
          expected = ArrayPartition(submanifold_components(G, prod)...)
          computed = compose(G, p1, p2)
          @test isapprox(G, expected, computed)
      end
  end
end

test_exp_ad(Random.default_rng(), MultiDisplacement(3, 2))

@testset "Test MultiAffine" for G in
    [
        MultiDisplacement(3, 2),
        MultiAffine(Unitary(4), 3),
    ]
    # begin
    test_multi_affine(Random.default_rng(), G)
end

