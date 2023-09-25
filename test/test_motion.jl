using Test
using GeometricFilter

using Manifolds

import LinearAlgebra

# Base.length(x::ProductRepr) = sum(map(Base.length, submanifold_components(x)))
# Base.convert(::Type{ArrayPartition}, x::ProductRepr) = ArrayPartition(submanifold_components(x)...)
# Base.convert(::Type{ProductRepr}, x::ArrayPartition) = ProductRepr(submanifold_components(x)...)

# TODO: test scalar * motion

@testset "Motion" begin
    # G = SpecialOrthogonal(4)
    G = MultiDisplacement(4,2)
    # G = SpecialEuclidean(4)
    action = GroupOperationAction(G)
    # vel = rand(G; vector_at=Identity(G))
    vel = rand(G; vector_at=identity_element(G))
    # submanifold_component(vel, 1) .= 0
    # rm = make_rigid_motion(action, vel)
    rm = RigidMotion(action, vel)
    rm + ZeroMotion(GroupOperationAction(G))
    @test_throws TypeError RigidMotion(GroupOperationAction(G,RightAction()), vel)
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

    # vel_ = rand(G; vector_at=Identity)
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

function test_constant_dict(d::Dict, comp)
    if length(d) == 0
        return
    end
    v_ = first(d).second
    @testset "Testing $k" for (k,v) in d
        @test comp(v_, v)
    end
end


@testset "Swap Group Motion" begin
    G = MultiDisplacement(3,2)
    ξ = rand(G; vector_at=Identity(G))

    # x0 = identity_element(G)
    x0 = rand(G)

    @testset "Rigid/Translation" begin

        R1 = RigidMotion(GroupOperationAction(G), ξ)
        T1 = TranslationMotion(G, ξ, RightAction())
        M1 = Dict(
            :R1 => R1,
            :R1_ => swap_group_motion(R1),
            :R1__ => GeometricFilter._swap_group_motion(R1),
            :T1 => T1,
            :T1_ => swap_group_motion(T1),
            :T1__ => GeometricFilter._swap_group_motion(T1),
        )

        rM1 = Dict([(s => integrate(v, x0)) for (s,v) in M1]...)
        # for v in values(rM1)
        #     @test isapprox(G, rM1[:R1], v)
        # end
        test_constant_dict(rM1, (a,b)->isapprox(G, a, b))

        R2 = RigidMotion(DualGroupOperationAction(G), ξ)
        T2 = TranslationMotion(G, ξ, LeftAction())
        M2 = Dict(
            :R2 => R2,
            :R2_ => swap_group_motion(R2),
            :R2__ => GeometricFilter._swap_group_motion(R2),
            :T2 => T2,
            :T2_ => swap_group_motion(T2),
            :T2__ => GeometricFilter._swap_group_motion(T2),
        )

        rM2 = Dict([(s => integrate(v, x0)) for (s,v) in M2]...)

        test_constant_dict(rM2, (a,b) -> isapprox(G, a, b))

    end

    @testset "Swap MultiAffineMotion" begin
        G = MultiDisplacement(3,2)
        m1 = MultiAffineMotion(G, [1.0 0;0 0], LeftAction())
        m2 = swap_group_motion(m1)
        MAM = Dict(
            :m1 => integrate(m1, x0),
            :m2 => integrate(m2, x0),
        )
        test_constant_dict(MAM, (a,b) -> isapprox(G, a, b))
    end

end

"""
Helper function used for testing flat affine motions.
"""
function flat_affine_motion(
    lin, trans,
    # B::AbstractBasis
)
    dim = size(lin, 2)
    G = TranslationGroup(dim)
    V = Euclidean(dim)
    A = TranslationAction(V, G)
    f(u) = lin * u + trans
    return AffineMotion(A, f, x -> lin * x)
end

@testset "Flat Motion" begin
    @testset "Scaling+Translation" begin
        dim = 2
        A = rand(dim, dim)
        b = rand(dim)
        m = flat_affine_motion(A, b)
        m_ = FlatAffineMotion(A, b)
        p = rand(dim)
        @test m * p ≈ m_ * p
    end
    @testset "Flat Translation" begin
        dim = 2
        t = rand(dim)
        m = RigidMotion(get_flat_action(dim), t)
        m_ = FlatAffineMotion(zeros(dim, dim), t)
        p = rand(dim)
        @test m * p ≈ m_ * p
    end
    @testset "Approx Equality" begin
        M1 = FlatAffineMotion(zeros(2,2), zeros(2))
        M2 = FlatAffineMotion(zeros(2,2), zeros(2))
        @test M1 ≈ M2
    end
end

@testset "Zero Motion" begin
    G = SpecialOrthogonal(3)
    M = Sphere(2)
    A = RotationAction(M, G)
    m = ZeroMotion(A)
    x = [1., 0, 0]
    @test m(x) ≈ zero_vector(G, identity_element(G))
    # @show m*x
    y = rand(M)
    @test isapprox(M, m'(x)(y), y)
end

"⚡"
