
struct TranslationMotion{TA, TG,TV,TAD<:ActionDirection} <: AbstractAffineMotion{TA}
    G::TG
    vel::TV # in Alg(G)
end

Base.show(io::IO, m::TranslationMotion) = print(io, "TranslationMotion($(m.G), $(m.vel))")

@doc raw"""
    TranslationMotion(G,vel,dir=LeftAction())

A translation motion defined on Lie groups.
There is a left and a right version.
The left version is associated to the left multiplication action of ``G`` on itself. The motion is equal to ``φ(χ) = -χξχ^{-1}``.
The right version is associated to the right multiplication action of ``G`` on itself. The motion is equal to ``φ(χ) = -χ^{-1}ξχ``.

In both cases, the linear part is the zero operator.
"""
TranslationMotion(G, vel, conv=LeftAction()) = TranslationMotion{typeof(_get_group_operation_action(G, conv)), typeof(G),typeof(vel),typeof(conv)}(G, vel)

# get_action(m::TranslationMotion{TG,TV,LeftAction}) where {TG,TV} = GroupOperationAction(m.G)
# get_action(m::TranslationMotion{TG,TV,RightAction}) where {TG,TV} = DualGroupOperationAction(m.G)
get_action(m::TranslationMotion{TA,TG,TV,TAD}) where {TA,TG,TV,TAD} = _get_group_operation_action(m.G, TAD())

get_dynamics(m::TranslationMotion{TA,TG,TV,LeftAction}, u) where {TA,TG,TV} =  -adjoint_action(m.G, u, m.vel)
# TODO: should be inverse adjoint action here?
get_dynamics(m::TranslationMotion{TA,TG,TV,RightAction}, u) where {TA,TG,TV} =  -adjoint_action(m.G, inv(m.G, u), m.vel)

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

