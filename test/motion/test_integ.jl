using Test
using GeometricFilter
using Manifolds

import LinearAlgebra
import Random

rng = Random.default_rng()

@testset "Motion $G" for G in [SpecialOrthogonal(4),
                            SpecialEuclidean(4),
                            MultiDisplacement(4),
                            MultiDisplacement(4,2),
                            ]
    action = GroupOperationAction(G)
    vel = rand(rng, GroupTools.algebra(G))
    # submanifold_component(vel, 1) .= 0
    # rm = make_rigid_motion(action, vel)
    rm = RigidMotion(action, vel)

    @testset "ZeroMotion" begin
        rm0 = rm + ZeroMotion(GroupOperationAction(G))
        @test rm0 isa RigidMotion
        @test rm0 ≈ rm
    end

    @testset "RigidMotion Exceptions" begin
        @test_throws ErrorException RigidMotion(action, 0)
        @test_throws TypeError RigidMotion(GroupOperationAction(G,(RightAction(), RightSide())), vel)
        # @show rm'(identity_element(G))
    end

    sol = GeometricFilter.integrate_lift(1.0*rm, identity_element(G), .01)
    # test that sol(1) ≡ exp(ξ), for a rigid motion ξ
    expected = exp_lie(G, vel)

    @testset "Integration" begin
        computed = last(sol)
        # @error G vel computed expected
        @test isapprox(G, computed, expected)

        rm_ = RigidMotion(action, -vel)
        sol_ = GeometricFilter.integrate_lift(rm+rm_, identity_element(G), .01)
        id = last(sol_)
        @test isapprox(G, id, identity_element(G))

        tm = TranslationMotion(G, vel, LeftAction())
        sol = GeometricFilter.integrate_lift(tm, identity_element(G), .1)
        # TODO: missing test here
    end

    # vel_ = rand(rng, GeometricFilter.algebra(G))
    # v3 = lie_bracket(G, vel, vel_)
    # @show vel_
    # @show adjoint_action(G, inv(G, χ), v3)

    @testset "compose adjoint" begin
        B = DefaultOrthogonalBasis()
        # mm = get_adjoint_matrix(G, vel, B)
        mm = GeometricFilter.get_lin_mat(rm, identity_element(G), B)
        res_mat = GeometricFilter.compose_adjoint(G, inv(G, expected), exp(mm), B)
        # display(res_mat)
        @test isapprox(res_mat, LinearAlgebra.I)
    end



end
