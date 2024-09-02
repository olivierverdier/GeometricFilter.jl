
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
observed_space(obs::ProductObserver) = observed_space(first(obs.observers))

get_measurement(obs::ProductObserver, x) = RecursiveArrayTools.ArrayPartition([ob(x) for ob in obs.observers]...)

get_tan_observer(obs::ProductObserver, A, x, ys) = ξ -> RecursiveArrayTools.ArrayPartition([get_tan_observer(ob, A, x, y)(ξ) for (ob, y) in zip(obs.observers, submanifold_components(ys))]...)
