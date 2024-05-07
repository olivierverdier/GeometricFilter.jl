

Manifolds.apply_diff_group(
    A::Manifolds.RotationActionOnVector{RightAction},
    Id::Identity,
    X,
    p,
) = -apply_diff_group(switch_direction(A), Id, X, p)

Manifolds.apply_diff_group(
    ::Manifolds.ColumnwiseMultiplicationAction{LeftAction},
    ::Identity,
    X,
    p,
) = X * p

Manifolds.apply!(::RotationAction{LeftAction}, q, a, p) = LinearAlgebra.mul!(q, a, p)

Manifolds.apply!(A::Manifolds.RotationAction{RightAction}, q, a, p) = Manifolds.apply!(switch_direction(A), q, inv(base_group(A), a), p)


Manifolds.apply_diff(A::Manifolds.ColumnwiseMultiplicationAction{LeftAction}, a, ::Any, X) = apply(A, a, X)


Manifolds.apply_diff_group!(::Manifolds.ColumnwiseMultiplicationAction{LeftAction}, Y, ::Identity, X, p) = LinearAlgebra.mul!(Y, X, p)
