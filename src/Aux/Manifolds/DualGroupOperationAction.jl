# Unfortunately, almost a copy of the code in Manifolds.jl

"""
    DualGroupOperationAction(G::DecoratedManifold)

The dual (left) group action of a group on itself, defined by
```math
a⋅b := b a^{-1}
```
"""
struct DualGroupOperationAction{G} <: AbstractGroupAction{LeftAction}
    group::G
end

Base.show(io::IO, A::DualGroupOperationAction) = print(io, "DualGroupOperationAction($(A.group), $(direction(A)))")

Manifolds.base_group(A::DualGroupOperationAction) = A.group

Manifolds.group_manifold(A::DualGroupOperationAction) = A.group


_get_group_operation_action(G, ::LeftAction) = GroupOperationAction(G, (LeftAction(), LeftSide()))
_get_group_operation_action(G, ::RightAction) = DualGroupOperationAction(G)


Manifolds.apply(A::DualGroupOperationAction, a, p) = translate(A.group, a, p, Manifolds.LeftBackwardAction())

Manifolds.apply!(A::DualGroupOperationAction, q, a, p) = inverse_translate!(A.group, q, a, p, Manifolds.LeftBackwardAction())

function Manifolds.inverse_apply(A::DualGroupOperationAction, a, p)
    return translate(A.group, a, p, Manifolds.RightBackwardAction())
end

function Manifolds.inverse_apply!(A::DualGroupOperationAction, q, a, p)
    return translate!(A.group, q, a, p, Manifolds.RightBackwardAction())
end

# ----
# function apply_diff(A::DualGroupOperationAction, a, p, X)
#     return translate_diff(A.group, a, p, X, direction(A))
# end

# function apply_diff!(A::DualGroupOperationAction, Y, a, p, X)
#     return translate_diff!(A.group, Y, a, p, X, direction(A))
# end

function Manifolds.apply_diff_group(
    A::DualGroupOperationAction{<:AbstractDecoratorManifold},
    id::Identity,
    ξ,
    p,
)
    G = base_group(A)
    return -translate_diff(G, p, id, ξ, Manifolds.LeftForwardAction())
end

function Manifolds.apply_diff_group!(
    A::DualGroupOperationAction{<:AbstractDecoratorManifold},
    Y,
    id::Identity,
    ξ,
    p,
) 
    G = base_group(A)
    return -translate_diff!(G, Y, p, id, ξ, Manifolds.LeftForwardAction())
end

# function inverse_apply_diff(A::DualGroupOperationAction, a, p, X)
#     return inverse_translate_diff(A.group, a, p, X, direction(A))
# end

# function inverse_apply_diff!(A::DualGroupOperationAction, Y, a, p, X)
#     return inverse_translate_diff!(A.group, Y, a, p, X, direction(A))
# end

