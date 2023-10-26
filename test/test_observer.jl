using Test
using GeometricFilter

using Manifolds
import Random: default_rng

import Base.Iterators

rng = default_rng()

@testset "Observer init" begin
    G = MultiDisplacement(2, 2)
    A = MultiAffineAction(G, [1.0, 0])
    # ref must be on the manifold:
    @test_throws ErrorException ActionObserver(A, zeros(3))
    @test observed_space(ActionObserver(A, [1, 0])) == G
    # O(identity_element(G))
end

@testset "observed_space" begin
    M = Sphere(2)
    obs = IdentityObserver(M)
    @test observed_space(obs) == M
end

@testset "Test Observers" begin
    G = MultiDisplacement(2, 2)
    M = G
    pos_obs = PositionObserver(MultiAffineAction(G, [1,0]))
    @test pos_obs(identity_element(G)) == zeros(2)
    ref = rand(2)
    obs = ActionObserver(MultiAffineAction(G, [1.0, 0]), ref)
    x = identity_element(G)
    @test obs(x) ≈ ref
    A = GroupOperationAction(G)
    H = GeometricFilter.get_tan_observer(obs, A, x)
    ξ = rand(rng, GeometricFilter.algebra(G))
    expected = apply_diff_group(get_action(obs), Identity(G), ξ, ref)
    computed = H(ξ)
    @test computed ≈ expected
    oobs = ProductObserver(obs, obs)
    @test observed_space(oobs) == G
    yy = oobs(x)
    @test first(yy.x) ≈ ref
    HH = GeometricFilter.get_tan_observer(oobs, A, x)
    @test first(HH(ξ).x) ≈ H(ξ)
end

"""
General test rules:
ob: G -> M, so if x = ob(χ), ob'(ξ)(x) is in T_xM
Left observation action ob(χ) = χ⋅x₀
Right observation action ob(χ) = χ⁻¹⋅x₀
Standard group action
    Left observation action -> ξ⋅ob(χ) = ob'(ξ)(ob(χ))
    Right observation action -> -(χ⁻¹⋆ξ)⋅ob(χ) = ob'(χ⁻¹⋆ξ)(ob(χ))
Dual group action
    Left observation action -> -(χ⋆ξ)⋅ob(χ) = -ob'(χ⋆ξ)(ob(χ))
    Right observation action -> ξ⋅ob(χ) = -ob'(ξ)(ob(χ))
"""

function inf_action(O::ActionObserver, χ, ξ)
    obact = get_action(O)
    id = Identity(base_group(obact))
    return apply_diff_group(obact, id, ξ, O(χ))
end


"""
ξ⋅ob(χ)
"""
function compute_tan_action(
    ::GroupOperationAction{LeftAction},
    O::ActionObserver{LeftAction},
    χ,
    ξ
    )
    return inf_action(O, χ, ξ)
end

"""
-(χ⁻¹⋆ξ)⋅ob(χ)
"""
function compute_tan_action(
    A::GroupOperationAction{LeftAction},
    O::ActionObserver{RightAction},
    χ,
    ξ
    )
    G = base_group(A)
    return inf_action(O, χ, adjoint_action(G, inv(G, χ), ξ))
end

"""
-(χ⋆ξ)⋅ob(χ)
"""
function compute_tan_action(
    A::DualGroupOperationAction,
    O::ActionObserver{LeftAction},
    χ,
    ξ
    )
    G = base_group(A)
    return -inf_action(O, χ, adjoint_action(G, χ, ξ))
end

"""
ξ⋅ob(χ)
"""
function compute_tan_action(
    ::DualGroupOperationAction,
    O::ActionObserver{RightAction},
    χ,
    ξ
    )
    return -inf_action(O, χ, ξ)
end



@testset "Test tan_action" begin
    dim = 2
    G = MultiDisplacement(dim,1)
    V = submanifold(G, 1)
    χ = rand(G)
    ξ = rand(rng, GeometricFilter.algebra(G))
    A = MultiAffineAction(G)
    x = rand(dim)

    obs = ActionObserver(A, x)
    obs_ = ActionObserver(switch_direction(A), x)

    @test obs(χ) ≈ obs_(inv(G,χ))

    GA = GroupOperationAction(G)
    GA_ = DualGroupOperationAction(G)
    @assert is_point(G, χ)

    @testset "Tan action $ob, $act" for ob in [obs, obs_], act in [GA, GA_]
        computed = GeometricFilter.get_tan_observer(ob, act, χ)(ξ)
        expected = compute_tan_action(act, ob, χ, ξ)
        @test computed ≈ expected
    end
    # @show computed = GeometricFilter.get_tan_observer(obs, GA, χ)(ξ)
    # @show expected = compute_tan_action(GA, obs, χ, ξ)

    # @show computed = GeometricFilter.get_tan_observer(obs_, GA, χ)(ξ)
    # @show expected = compute_tan_action(GA, obs_, χ, ξ)

    # @show expected = compute_tan_action(GA_, obs, χ, ξ)
    # @show computed = GeometricFilter.get_tan_observer(obs, GA_, χ, ξ)

    # @show expected = compute_tan_action(GA_, obs_, χ, ξ)
    # @show computed = GeometricFilter.get_tan_observer(obs_, GA_, χ, ξ)
    

    # GA = GroupOperationAction(G)
    # obs = PositionObserver(A)
    # @show obs(x)
    # obs_ = ActionObserver(switch_direction(A), [1., 0.])
    # @show obs_(x)
    # op = GeometricFilter.get_tan_observer(obs, GA, x)
    # ξ = rand(rng, GeometricFilter.algebra(G))
    # @show op(ξ)
    # H = GeometricFilter.get_op_matrix(G, GeometricFilter.get_manifold(obs), obs(x), op, DefaultOrthonormalBasis(), DefaultOrthonormalBasis())
    # @show H
end

_repr(::LeftAction) = "→"
_repr(::RightAction) = "←"


@testset "Linear $(_repr(conv)) $depth" for conv in [LeftAction(), RightAction()], depth in [0, 4]
    lobs = LinearObserver(zeros(2,3), conv; depth=depth)
    x = rand(observed_space(lobs))
    @show lobs
    res = lobs(x)
    @test is_point(observation_space(lobs), res)
end
