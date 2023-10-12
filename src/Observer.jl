
abstract type AbstractObserver end


#--------------------------------
# Interface
#--------------------------------
"""
    observation_space(obs::AbstractObserver) :: AbstractManifold

The manifold on which the measurement lies.
"""
function observation_space end

@deprecate get_manifold(obs::AbstractObserver) observation_space(obs)

"""
    observation_space(obs::AbstractObserver) :: AbstractManifold

The manifold which is observed
"""
function observed_space end

"""
   get_measurement(::AbstractObserver, x) :: [point in observation manifold]

The actual observer function, mapping a point ``x`` in the state space to a point in the measurement manifold.
"""
function get_measurement end


"""
    get_tan_observer(
        ::AbstractObserver,
        A, # group action G ⊂ Diff(M)
        x, # point in M
        y=obs(x), # point in N
    ) :: Function

Return the pullback of the tangent map of the observer, that is
the function defined by ``H(ξ) = ⟨dh, ξ⋅x⟩``.
"""
function get_tan_observer end
#--------------------------------

get_tan_observer(obs::AbstractObserver, A, x) = get_tan_observer(obs, A, x, obs(x))

(obs::AbstractObserver)(x) = get_measurement(obs, x)



include("Observer/Product.jl")

include("Observer/Simple/Linear.jl")
include("Observer/Simple/Action.jl")
include("Observer/Simple/Identity.jl")



