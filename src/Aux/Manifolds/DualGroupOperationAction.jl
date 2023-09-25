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

function Base.show(io::IO, A::DualGroupOperationAction)
    return print(io, "DualGroupOperationAction($(A.group), $(direction(A)))")
end

Manifolds.base_group(A::DualGroupOperationAction) = A.group

Manifolds.group_manifold(A::DualGroupOperationAction) = A.group


_get_group_operation_action(G, ::LeftAction) = GroupOperationAction(G, LeftAction())
_get_group_operation_action(G, ::RightAction) = DualGroupOperationAction(G)




Manifolds.apply(A::DualGroupOperationAction, a, p) = inverse_translate(A.group, a, p, RightAction())

Manifolds.apply!(A::DualGroupOperationAction, q, a, p) = inverse_translate!(A.group, q, a, p, RightAction())

function Manifolds.inverse_apply(A::DualGroupOperationAction, a, p)
    return translate(A.group, a, p, RightAction())
end

function Manifolds.inverse_apply!(A::DualGroupOperationAction, q, a, p)
    return translate!(A.group, q, a, p, RightAction())
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
    return -translate_diff(G, p, id, ξ, LeftAction())
end

function Manifolds.apply_diff_group!(
    A::DualGroupOperationAction{<:AbstractDecoratorManifold},
    Y,
    id::Identity,
    ξ,
    p,
) 
    G = base_group(A)
    return -translate_diff!(G, Y, p, id, ξ, LeftAction())
end

# function inverse_apply_diff(A::DualGroupOperationAction, a, p, X)
#     return inverse_translate_diff(A.group, a, p, X, direction(A))
# end

# function inverse_apply_diff!(A::DualGroupOperationAction, Y, a, p, X)
#     return inverse_translate_diff!(A.group, Y, a, p, X, direction(A))
# end

