
"""
    swap_group_motion(::Motion)

Swap from standard to dual group action, while preserving the same dynamics.
"""
swap_group_motion(m::RigidMotion{<:GroupOperationAction{LeftAction, LeftSide}}) = TranslationMotion(base_group(get_action(m)), m.vel, RightAction())
swap_group_motion(m::TranslationMotion{RightAction}) = RigidMotion(GroupOperationAction(m.G), m.vel)
swap_group_motion(m::RigidMotion{<:GroupOperationAction{LeftAction, RightSide}}) = TranslationMotion(base_group(get_action(m)), m.vel, LeftAction())
swap_group_motion(m::TranslationMotion{LeftAction}) = RigidMotion(DualGroupOperationAction(m.G), m.vel)

# _swap_inv(::GroupOperationAction, G, χ) = inv(G,χ)
# _swap_inv(::GroupOperationAction{LeftAction, RightSide}, G, χ) = χ
_swap_adjoint_action(::GroupOperationAction{LeftAction, LeftSide}, G, χ, ξ) = GU.inverse_adjoint_action(G, χ, ξ)
_swap_adjoint_action(::GroupOperationAction{LeftAction, RightSide}, G, χ, ξ) = adjoint_action(G, χ, ξ)
_swap_group_action(A::GroupOperationAction{LeftAction, LeftSide}) = DualGroupOperationAction(base_group(A))
_swap_group_action(A::GroupOperationAction{LeftAction, RightSide}) = GroupOperationAction(base_group(A))

