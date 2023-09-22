
Manifolds.apply!(A::Manifolds.RotationAction{TM,TG,RightAction}, q, a, p) where {TM,TG} = Manifolds.apply!(switch_direction(A), q, inv(base_group(A),a), p)

function Manifolds.apply_diff_group(
    A::Manifolds.RotationActionOnVector{N,F,RightAction},
    Id::Identity,
    X,
    p,
) where {N,F}
    return -apply_diff_group(switch_direction(A), Id, X, p)
end

function Manifolds.apply_diff_group(
    ::Manifolds.ColumnwiseMultiplicationAction{N,F,LeftAction},
    ::Identity,
    X,
    p,
) where {N,F}
    return X * p
end
