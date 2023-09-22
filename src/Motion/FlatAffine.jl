
@doc raw"""
    FlatAffineMotion(A::Matrix,b::Vector)

Creates the affine motion corresponding to the vector field ``x ↦ A x + b``.

```jldoctest
A = [0 1.;0 0]
b = [0, 0]
motion = FlatAffineMotion(A,b)

x0 = [2., 10] # position zero, velocity 10
x1 = integrate(motion, x0)
# output
[12., 10]
```
"""
struct FlatAffineMotion{TM,TV} <: AbstractAffineMotion
    linear::TM
    translation::TV
end

"""
    get_flat_action(d)

The action associated to a flat space of dimension ``d``,
namely a translation group acting on a Euclidean space.
"""
function get_flat_action(dim)
	  G = TranslationGroup(dim)
    V = Euclidean(dim)
    return TranslationAction(V,G)
end

get_action(m::FlatAffineMotion) = get_flat_action(size(m.translation, 1))

get_dynamics(m::FlatAffineMotion, x) = m.linear*x + m.translation

get_lin(m::FlatAffineMotion) = x -> m.linear*x


Base.isapprox(M1::FlatAffineMotion, M2::FlatAffineMotion; kwargs...) = isapprox(M1.linear, M2.linear; kwargs...) && isapprox(M1.translation, M2.translation; kwargs...)
