
"""
    ActionObserver(A::AbstractGroupAction{<:ActionDirection}, ref)

Models the observation of a Lie group element χ as the action
of χ on some manifold N.

If it is a *left* action, the corresponding observation is χ ↦ χ ⋅ x₀, where x₀ ∈ N is a reference point. This often corresponds to a measurement of the state, for instance, the position.

If it is a *right* action, the corresponding observation is ``χ ↦ χ^{-1} ⋅ x₀``, where ``x₀ ∈ N`` is a reference point. This often corresponds to the distance and bearing measurement of a fixed landmark.
"""
struct ActionObserver{TAD<:ActionDirection, TA<:AbstractGroupAction{TAD}} <: AbstractObserver
    action::TA  # action G ⊂ Diff(N)
    ref # point in N

    ActionObserver(A::AbstractGroupAction{TAD}, ref) where {TAD} = is_point(group_manifold(A),ref) ? new{TAD, typeof(A)}(A,ref) : error("ref must be in manifold")
end

Base.show(io::IO, obs::ActionObserver) = print(io, "ActionObserver($(obs.action), $(obs.ref))")

get_action(obs::ActionObserver) = obs.action
get_ref(obs::ActionObserver) = obs.ref
observation_space(obs::ActionObserver) = group_manifold(get_action(obs))

get_measurement(obs::ActionObserver, χ) = apply(get_action(obs), χ, get_ref(obs))

PositionObserver(A::MultiAffineAction{TH,dim,size,𝔽,LeftAction}) where {TH,dim,size,𝔽} = ActionObserver(A, zeros(dim))




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
Translate infinitesimal action into the
left action of the observation group on itself,
in other words, we solve the equation ξ⋅χ = η⋅χ
and return η.
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
    # compute η = v χ⁻¹, or χ⁻¹ v depending on observation
    # action direction
    η_ = inverse_translate_diff(G, χ, χ, v, switch_direction(AD()))
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
