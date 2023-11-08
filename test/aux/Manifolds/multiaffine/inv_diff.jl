using Test
using Manifolds
using Random
rng = Random.default_rng()

include("utils.jl")

"""
Test the differential of the inverse on a Lie group.
Denote this inverse by ``I(g) := g^{-1}``.
If the left and right transports are ``T_L(g,ξ) := gξ``
and ``T_R(g,ξ) := ξg`` respectively, then
```math
⟨DI(g), T_L(g,ξ)⟩ = -T_R(g^{-1}, ξ)
```
and
``` math
⟨DI(g), T_R(g,ξ)⟩ = -T_L(g^{-1}, ξ)
```
"""
test_inv_diff(G, χ, ξ, conv) = begin
    χ_ = inv(G, χ)
    @test isapprox(TangentSpace(G, χ_), inv_diff(G, χ, move(G, χ, ξ, conv)), -move(G, χ_, ξ, switch_side(conv)))
end

test_inv_diff_set(G, χ, ξ) = @testset "inv_diff" for side in [LeftSide(), RightSide()]
    test_inv_diff(G, χ, ξ, side)
end


@testset "inv_diff" begin
    G = MultiDisplacement(3,2)
    χ = rand(rng, G)
    ξ = rand(rng, TangentSpace(G, identity_element(G)))
    test_inv_diff_set(G, χ, ξ)
end
