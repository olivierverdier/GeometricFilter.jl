
"""
    swap_group_motion(::Motion)

Swap from standard to dual group action, while preserving the same dynamics.
"""
swap_group_motion(m::RigidMotion{<:GroupOperationAction}) = TranslationMotion(base_group(get_action(m)), m.vel, RightAction())
swap_group_motion(m::TranslationMotion{RightAction}) = RigidMotion(GroupOperationAction(m.G), m.vel)
swap_group_motion(m::RigidMotion{<:DualGroupOperationAction}) = TranslationMotion(base_group(get_action(m)), m.vel, LeftAction())
swap_group_motion(m::TranslationMotion{LeftAction}) = RigidMotion(DualGroupOperationAction(m.G), m.vel)

# _swap_inv(::GroupOperationAction, G, χ) = inv(G,χ)
# _swap_inv(::DualGroupOperationAction, G, χ) = χ
_swap_adjoint_action(::GroupOperationAction, G, χ, ξ) = inverse_adjoint_action(G, χ, ξ)
_swap_adjoint_action(::DualGroupOperationAction, G, χ, ξ) = adjoint_action(G, χ, ξ)
_swap_group_action(A::GroupOperationAction) = DualGroupOperationAction(base_group(A))
_swap_group_action(A::DualGroupOperationAction) = GroupOperationAction(base_group(A))

