
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
struct FlatAffineMotion{TA,TM,TV} <: SimpleAffineMotion{TA}
    linear::TM
    translation::TV
end


FlatAffineMotion(linear, translation) = FlatAffineMotion{typeof(_get_flat_action_from_translation(translation)), typeof(linear), typeof(translation)}(linear, translation)

Base.show(io::IO, m::FlatAffineMotion) = print(io, "FlatAffineMotion($(m.linear), $(m.translation))")

rescale_motion(s::Number, m::FlatAffineMotion) = FlatAffineMotion(s*m.linear, s*m.translation)

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

_get_flat_action_from_translation(translation) = get_flat_action(size(translation, 1))

get_action(m::FlatAffineMotion) = _get_flat_action_from_translation(m.translation)

get_dynamics(m::FlatAffineMotion, x) = m.linear*x + m.translation

get_lin(m::FlatAffineMotion) = x -> m.linear*x


Base.:+(lms::FlatAffineMotion{TA}...)  where {TA} = _add_flataffine_motions(lms...)

function _add_flataffine_motions(lms::FlatAffineMotion{TA}...)  where {TA}
    return FlatAffineMotion(sum([lm.linear for lm in lms]), sum([lm.translation for lm in lms]))
end

Base.isapprox(M1::FlatAffineMotion, M2::FlatAffineMotion; kwargs...) = isapprox(M1.linear, M2.linear; kwargs...) && isapprox(M1.translation, M2.translation; kwargs...)
