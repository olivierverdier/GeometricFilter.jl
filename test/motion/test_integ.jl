using Test
using GeometricFilter
using Manifolds

import LinearAlgebra
import Random

rng = Random.default_rng()

@testset "Motion" for G in [SpecialOrthogonal(4),
                            # SpecialEuclidean(4),
                            MultiDisplacement(4,2),
                            ]
    # G = SpecialOrthogonal(4)
    # G = MultiDisplacement(4,2)
    # G = SpecialEuclidean(4)
    action = GroupOperationAction(G)
    vel = rand(rng, GroupTools.algebra(G))
    # submanifold_component(vel, 1) .= 0
    # rm = make_rigid_motion(action, vel)
    rm = RigidMotion(action, vel)
    @test_throws ErrorException RigidMotion(action, 0)
    rm0 = rm + ZeroMotion(GroupOperationAction(G))
    @test rm0 isa RigidMotion
    @test rm0 ≈ rm

    @test_throws TypeError RigidMotion(GroupOperationAction(G,(RightAction(), RightSide())), vel)
    # @show rm'(identity_element(G))
    sol = GeometricFilter.integrate_lift(1.0*rm, identity_element(G), .01)
    # test that sol(1) ≡ exp(ξ), for a rigid motion ξ
    χ = sol(1)
    expected = exp_lie(G, vel)
    @test isapprox(G, χ, expected)

    rm_ = RigidMotion(action, -vel)
    sol_ = GeometricFilter.integrate_lift(rm+rm_, identity_element(G), .01)
    id = sol_(1)
    @test isapprox(G, id, identity_element(G))

    tm = TranslationMotion(G, vel, LeftAction())
    sol = GeometricFilter.integrate_lift(tm, identity_element(G), .1)

    # vel_ = rand(rng, GeometricFilter.algebra(G))
    # v3 = lie_bracket(G, vel, vel_)
    # @show vel_
    # @show adjoint_action(G, inv(G, χ), v3)

    B = DefaultOrthogonalBasis()
    # mm = get_adjoint_matrix(G, vel, B)
    mm = GeometricFilter.get_lin_mat(rm, identity_element(G), B)
    res_mat = GeometricFilter.compose_adjoint(G, inv(G, χ), exp(mm), B)


    # display(res_mat)
    @test isapprox(res_mat, LinearAlgebra.I)

end
