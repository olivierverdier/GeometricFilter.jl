
const MultiDisplacement{dim,size} = MultiAffine{
    SpecialOrthogonal{ManifoldsBase.TypeParameter{Tuple{dim}}},
    dim,
    size,
    ℝ
}

"""
    MultiDisplacement(n, size=1)

A special case of the [`MultiAffine`](@ref) group, where the
underlying group is the special orthogonal group ``SO(n)``.
"""
MultiDisplacement(dim::Integer, size::Integer=1) = MultiAffine(SpecialOrthogonal(dim), size)

Base.show(io::IO, ::MultiDisplacement{dim,size}) where {dim,size} = print(io, "MultiDisplacement($(dim), $(size))")

Base.show(io::IO, ::MultiDisplacement{dim,1}) where {dim} = print(io, "MultiDisplacement($(dim))")
