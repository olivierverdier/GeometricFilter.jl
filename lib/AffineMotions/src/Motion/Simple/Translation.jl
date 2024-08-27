_get_group_operation_action(G, side) = GroupOperationAction(G, (LeftAction(), side))

struct TranslationMotion{TAD<:Manifolds.GroupActionSide,TA,TG,TV} <: SimpleAffineMotion{TA}
    G::TG
    vel::TV # in Alg(G)
end

Base.show(io::IO, m::TranslationMotion) = print(io, "TranslationMotion($(m.G), $(m.vel))")

rescale_motion(s::Number, m::TranslationMotion{TAD}) where {TAD} = TranslationMotion(m.G, s*m.vel, TAD())

@doc raw"""
    TranslationMotion(G,vel,side=LeftSide())

A translation motion defined on Lie groups.
There is a left and a right version.
The left version is associated to the left multiplication action of ``G`` on itself. The motion is equal to ``φ(χ) = -χξχ^{-1}``.
The right version is associated to the right multiplication action of ``G`` on itself. The motion is equal to ``φ(χ) = -χ^{-1}ξχ``.

In both cases, the linear part is the zero operator.
"""
TranslationMotion(G, vel, conv::Manifolds.GroupActionSide) = TranslationMotion{typeof(conv),typeof(_get_group_operation_action(G, conv)),typeof(G),typeof(vel)}(G, vel)

get_action(m::TranslationMotion{TAD}) where {TAD} = _get_group_operation_action(m.G, TAD())

get_dynamics(m::TranslationMotion{LeftSide}, u) =  -adjoint_action(m.G, u, m.vel)
# TODO: should be inverse adjoint action here?
get_dynamics(m::TranslationMotion{RightSide}, u) =  -adjoint_action(m.G, inv(m.G, u), m.vel)

function get_lin(m::TranslationMotion)
    G = base_group(get_action(m))
    zero = zero_vector(G, identity_element(G))
    function lin(ξ)
        return zero
        # res = similar(ξ)
        # fill!(res, 0.)
        # return res
    end
    return lin
end


Base.:+(m1::TranslationMotion{<:Any, TA}, m2::TranslationMotion{<:Any, TA})  where {TA} = _add_translation_motions(m1,m2)

function _add_translation_motions(m1::TranslationMotion{TAD}, m2::TranslationMotion{TAD}) where {TAD}
    return TranslationMotion(m1.G, m1.vel+m2.vel, TAD())
end

Base.isapprox(M1::TranslationMotion{<:Any, TA}, M2::TranslationMotion{<:Any, TA}; kwargs...) where {TA} = isapprox(M1.vel, M2.vel; kwargs...)
