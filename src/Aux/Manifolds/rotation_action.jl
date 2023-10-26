

function Manifolds.apply_diff_group(
    A::Manifolds.RotationActionOnVector{RightAction},
    Id::Identity,
    X,
    p,
) 
    return -apply_diff_group(switch_direction(A), Id, X, p)
end

function Manifolds.apply_diff_group(
    ::Manifolds.ColumnwiseMultiplicationAction{LeftAction},
    ::Identity,
    X,
    p,
)
    return X * p
end

Manifolds.apply!(::RotationAction{LeftAction}, q, a, p)  = LinearAlgebra.mul!(q, a, p)

Manifolds.apply!(A::Manifolds.RotationAction{RightAction}, q, a, p) = Manifolds.apply!(switch_direction(A), q, inv(base_group(A), a), p)
