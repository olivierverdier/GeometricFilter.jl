using Test
using Manifolds
using GeometricFilter
using Random
rng = Random.default_rng()

include("utils.jl")

_get_side(::LeftAction) = RightSide()
_get_side(::RightAction) = LeftSide()


_transporter(G, χ, ξ, dir) = move(G, χ, ξ, _get_side(dir))

"""
This should hold for *any* group action ``A`` on any manifold.
If you define ``π_x(g) := A(g, x)`` for ``g ∈ G`` and ``x ∈ M``,
and define, for ``ξ in Alg(G)``,
 ``T_R(g, ξ) := ξg`` (the right translation),
and ``T_L(g, ξ) := gξ`` (the left translation), then we have the identity:
```math
⟨Dπ_{x}(g), T(g, ξ)⟩ = ⟨Dπ_{A(g,x)}(1), ξ⟩
```
where, for a *left* action, ``T`` is the *right* translation,
and for a *right* action, ``T`` is the *left* translation.
"""
test_apply_diff_group(A::AbstractGroupAction{TAD}, χ, ξ, p) where {TAD} = begin
    G = base_group(A)
    p_ = apply(A, χ, p)
    v1 = apply_diff_group(A, χ, _transporter(G, χ, ξ, TAD()), p)
    v2 = apply_diff_group(A, identity_element(G), ξ, p_)
    @test isapprox(TangentSpace(G, p_), v1, v2)
end


test_apply_diff_group_set(G, χ, ξ, χ_) = begin
    @testset "$dir, $side" for side in [LeftSide(), RightSide()], dir in [LeftAction(), RightAction()]
            test_apply_diff_group(GroupOperationAction(G, (dir, side)), χ, ξ, χ_)
    end
end


@testset "apply_diff_group" begin
    G = MultiDisplacement(3,2)
    # χ1 = rand(rng, G)
    χ1 = Identity(G)
    χ2 = rand(rng, G)
    ξ = rand(rng, TangentSpace(G, identity_element(G)))
    test_apply_diff_group_set(G, χ1, ξ, χ2)
end
