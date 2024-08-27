using Test
using Manifolds

function test_constant_dict(d::Dict, comp)
    if length(d) == 0
        return
    end
    v_ = first(d).second
    @testset "Testing $k" for (k, v) in d
        @test comp(v_, v)
    end
end

@testset "Swap Group Motion $G" for G in [
    SpecialOrthogonal(3),
    # SpecialEuclidean(3),
    MultiDisplacement(3, 1),
    MultiDisplacement(3, 2),
    ]
    ξ = rand_lie(rng, G)

    # x0 = identity_element(G)
    x0 = rand(rng, G)

    @testset "Rigid/Translation" begin

        R1 = RigidMotion(GroupOperationAction(G), ξ)
        T1 = TranslationMotion(G, ξ, RightAction())
        M1 = Dict(
            :R1 => R1,
            :R1_ => swap_group_motion(R1),
            :R1__ => AffineMotions._swap_group_motion(R1),
            :T1 => T1,
            :T1_ => swap_group_motion(T1),
            :T1__ => AffineMotions._swap_group_motion(T1),
        )

        rM1 = Dict([(s => integrate(v, x0)) for (s,v) in M1]...)
        # for v in values(rM1)
        #     @test isapprox(G, rM1[:R1], v)
        # end
        test_constant_dict(rM1, (a,b)->isapprox(G, a, b))

        # R2 = RigidMotion(DualGroupOperationAction(G), ξ)
        R2 = RigidMotion(GroupOperationAction(G, (LeftAction(), RightSide())), ξ)
        T2 = TranslationMotion(G, ξ, LeftAction())
        M2 = Dict(
            :R2 => R2,
            :R2_ => swap_group_motion(R2),
            :R2__ => AffineMotions._swap_group_motion(R2),
            :T2 => T2,
            :T2_ => swap_group_motion(T2),
            :T2__ => AffineMotions._swap_group_motion(T2),
        )

        rM2 = Dict([(s => integrate(v, x0)) for (s,v) in M2]...)

        test_constant_dict(rM2, (a,b) -> isapprox(G, a, b))

    end


end


@testset "Swap AdjointLinearMotion" begin
    G = MultiDisplacement(3,2)
    x0 = rand(rng, G)
    m1 = AdjointLinearMotion(G, [1.0 0;0 0], LeftAction())
    m2 = swap_group_motion(m1)
    MAM = Dict(
        :m1 => integrate(m1, x0),
        :m2 => integrate(m2, x0),
    )
    test_constant_dict(MAM, (a,b) -> isapprox(G, a, b))
end

@testset "Swap AdjointLinear sum" begin
    G = MultiDisplacement(3,2)
    ξ = rand_lie(rng, G)
    m1 = AdjointLinearMotion(G, [1.0 0;0 0], LeftAction())
    rm = RigidMotion(GroupOperationAction(G), ξ)
    @test swap_group_motion(m1+rm) isa AffineMotions.AffineMotionSum
end
