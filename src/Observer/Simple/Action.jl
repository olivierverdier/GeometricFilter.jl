
"""
    ActionObserver(A::AbstractGroupAction{<:ActionDirection}, ref)

Models the observation of a Lie group element χ as the action
of χ on some manifold N.

If it is a *left* action, the corresponding observation is χ ↦ χ ⋅ x₀, where x₀ ∈ N is a reference point. This often corresponds to a measurement of the state, for instance, the position.

If it is a *right* action, the corresponding observation is ``χ ↦ χ^{-1} ⋅ x₀``, where ``x₀ ∈ N`` is a reference point. This often corresponds to the distance and bearing measurement of a fixed landmark.
"""
struct ActionObserver{TAD<:ActionDirection, TA<:AbstractGroupAction{TAD}, TV} <: AbstractObserver
    action::TA  # action G ⊂ Diff(N)
    ref::TV # point in N

    ActionObserver(A::AbstractGroupAction{TAD}, ref) where {TAD} = is_point(group_manifold(A),ref) ? new{TAD, typeof(A), typeof(ref)}(A,ref) : error("ref must be in manifold")
end

Base.show(io::IO, obs::ActionObserver) = print(io, "ActionObserver($(obs.action), $(obs.ref))")

get_action(obs::ActionObserver) = obs.action
get_ref(obs::ActionObserver) = obs.ref
observation_space(obs::ActionObserver) = group_manifold(get_action(obs))
observed_space(obs::ActionObserver) = base_group(get_action(obs))

get_measurement(obs::ActionObserver, χ) = apply(get_action(obs), χ, get_ref(obs))

PositionObserver(A::MultiAffineAction{LeftAction, <:MultiAffine{<:Any, dim}}) where {dim} = ActionObserver(A, zeros(dim))





function tan_observation(
    obs::ActionObserver,
    pred, # in observation manifold
    η, # in observation Lie algebra
    )
    action = get_action(obs)
    G = base_group(action)
    tvec = apply_diff_group(action, Identity(G), η, pred)
    return tvec
end

# switch_sign(::LeftAction) = 1.
# switch_sign(::RightAction) = -1.

"""
Compute ``η = v χ⁻¹``, or ``χ⁻¹ v`` depending on whether the observation action direction is `Left` or `Right` respectively.
The previous version of `Manifolds.jl` (< v0.9.0) allowed to write:
```julia
η_ = inverse_translate_diff(G, χ, χ, v, switch_direction(AD()))
```
For the new version (≥ v0.9.0), we use that
``v χ⁻¹`` is obtained by `translate_diff(G, inv(G,χ), χ, v, RightBackwardAction())`
and ``χ⁻¹ v`` is obtained by `translate_diff(G, inv(G,χ), χ, v, LeftForwardAction())`.
"""
_translate_to_identity(G, χ, v, ::LeftAction) = translate_diff(G, inv(G,χ), χ, v, Manifolds.RightBackwardAction())
_translate_to_identity(G, χ, v, ::RightAction) = translate_diff(G, inv(G, χ), χ, v, Manifolds.LeftForwardAction())

"""
Translate infinitesimal action into the
left action of the observation group on itself,
in other words, we solve the equation ``ξ⋅χ = η∘χ``
and return ``η``, where ``∘`` denotes the
standard or dual left action of the group on itself.
"""
function translate_action(
    obs::ActionObserver{AD}, # action G ⊂ Diff(M) OR G* ⊂ Diff(M)
    action::AbstractGroupAction{LeftAction}, # action H ⊂ Diff(G)
    χ, # point in G
    ξ, # in Alg(H)
    ) where {AD<:ActionDirection}
    H = base_group(action)
    G = base_group(get_action(obs))
    # tangent vector from infinitesimal action:
    v = apply_diff_group(action, Identity(H), ξ, χ)

    # compute η = v χ⁻¹, or χ⁻¹ v depending on observation action direction
    # η_ = inverse_translate_diff(G, χ, χ, v, switch_direction(AD()))
    η_ =_translate_to_identity(G, χ, v, AD())

    # η = switch_sign(AD()) * η_
    η = η_
    return η
end


function get_tan_observer(
    obs::ActionObserver,
    action::AbstractGroupAction{LeftAction}, # an action of a group on the observation group
    χ, # point in observation group
    pred,
    )
    function tan_observer(ξ)
        η = translate_action(obs, action, χ, ξ)
        res = tan_observation(obs, pred, η)
        return res
    end
    return tan_observer
end


# TODO: improve performance by directly compute get_obs_matrix in some particular cases?
