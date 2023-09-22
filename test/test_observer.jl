using Test
using GeometricFilter

using Manifolds
import Random: default_rng

import Base.Iterators
import Printf

rng = default_rng()

@testset "Observer init" begin
    G = MultiDisplacement(2,2)
    A = MultiAffineAction(G, [1.,0])
    @test_throws ErrorException ActionObserver(A, zeros(3))
    # O(identity_element(G))
end

@testset "Test Observers" begin
    G = MultiDisplacement(2,2)
    M = G
    A = GroupOperationAction(G)
    # obs = PositionObserver(MultiAffineAction(G))
    obs = ActionObserver(MultiAffineAction(G, [1.,0]), [1., 0])
    x = identity_element(G)
    @show obs(x)
    H = GeometricFilter.get_tan_observer(obs, A, x)
    # @infiltrate
    @show H(x)
    oobs = ProductObserver(obs, obs)
    @show oobs(x)
    HH = GeometricFilter.get_tan_observer(oobs, A, x)
    @show HH(x)
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
    A::GroupOperationAction{TG,LeftAction},
    O::ActionObserver{LeftAction},
    χ,
    ξ
    ) where{TG}
    return inf_action(O, χ, ξ)
end

"""
-(χ⁻¹⋆ξ)⋅ob(χ)
"""
function compute_tan_action(
    A::GroupOperationAction{TG,LeftAction},
    O::ActionObserver{RightAction},
    χ,
    ξ
    ) where {TG}
    G = base_group(A)
    return inf_action(O, χ, adjoint_action(G, inv(G, χ), ξ))
end

"""
-(χ⋆ξ)⋅ob(χ)
"""
function compute_tan_action(
    A::DualGroupOperationAction{TG},
    O::ActionObserver{LeftAction},
    χ,
    ξ
    ) where{TG}
    G = base_group(A)
    return -inf_action(O, χ, adjoint_action(G, χ, ξ))
end

"""
ξ⋅ob(χ)
"""
function compute_tan_action(
    A::DualGroupOperationAction{TG},
    O::ActionObserver{RightAction},
    χ,
    ξ
    ) where {TG}
    return -inf_action(O, χ, ξ)
end



@testset "Test tan_action" begin
    dim = 2
    G = MultiDisplacement(dim,1)
    V = submanifold(G, 1)
    χ = rand(G)
    ξ = rand(G; vector_at=Identity(G))
    A = MultiAffineAction(G)
    x = rand(dim)

    obs = ActionObserver(A, x)
    obs_ = ActionObserver(switch_direction(A), x)

    @show obs(χ)
    @show obs_(χ)

    GA = GroupOperationAction(G)
    GA_ = DualGroupOperationAction(G)
    @assert is_point(G, χ)

    for (ob, act) in Base.Iterators.product([obs, obs_], [GA, GA_])
        computed = GeometricFilter.get_tan_observer(ob, act, χ)(ξ)
        expected = compute_tan_action(act, ob, χ, ξ)
        @test computed ≈ expected
        # Printf.@printf "\n👉\t%s\n👀\t%s\n" act ob
        # @show expected
        # @show computed
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
    # ξ = rand(G; vector_at=Identity(G))
    # @show op(ξ)
    # H = GeometricFilter.get_op_matrix(G, GeometricFilter.get_manifold(obs), obs(x), op, DefaultOrthonormalBasis(), DefaultOrthonormalBasis())
    # @show H
end
