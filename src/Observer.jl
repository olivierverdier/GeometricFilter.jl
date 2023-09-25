
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



struct ProductObserver{TO<:Tuple} <: AbstractObserver
    observers::TO
end

Base.show(io::IO, obs::ProductObserver) = print(io, "ProductObserver("* join([repr(o) for o in obs.observers], ", ") *")")

"""
    ProductObserver(observers::AbstractObserver...)

Combine existing observers into a new one.

```julia
M = ManifoldsBase.DefaulManifold(2)
# a simple observer for the sake of this example
ob1 = IdentityObserver(M)
ob2 = IdentityObserver(M)
x = rand(M) # a point on M
ob = ProductObserver(ob1, ob2)
ob(x) # (x, x)
```
"""
function ProductObserver(observers::AbstractObserver...)
    return ProductObserver(observers)
end

observation_space(obs::ProductObserver) = ProductManifold([observation_space(o) for o in obs.observers]...)

get_measurement(obs::ProductObserver, x) = ArrayPartition([ob(x) for ob in obs.observers]...)

get_tan_observer(obs::ProductObserver, A, x, ys) = ξ -> ArrayPartition([get_tan_observer(ob, A, x, y)(ξ) for (ob,y) in zip(obs.observers, submanifold_components(ys))]...)

include("Observer/Linear.jl")
include("Observer/Action.jl")
include("Observer/Identity.jl")



