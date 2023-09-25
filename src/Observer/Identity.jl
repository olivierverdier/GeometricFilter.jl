"""
    IdentityObserver(M)

Direct observation of a point on the state manifold.

```julia
M = Sphere(2)
x = rand(M)
obs = IdentityObserver(M)
obs(x) # x
```
"""
struct IdentityObserver{TM} <: AbstractObserver
    M::TM # the manifold on which observation is done
end

Base.show(io::IO, obs::IdentityObserver) = print(io, "IdentityObserver($(obs.M))")

observation_space(obs::IdentityObserver) = obs.M

get_measurement(::IdentityObserver, x) =  x

get_tan_observer(::IdentityObserver, A::AbstractGroupAction{LeftAction}, x, y) = ξ -> apply_diff_group(A, Identity(base_group(A)), ξ, x)

