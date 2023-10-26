
struct LinearObserver{TAD, TV, TI} <: AbstractObserver
    matrix::TV
    depth::TI
end

@doc raw"""
    LinearObserver(H::AbstractMatrix,
      conv=LeftAction();
      depth=0)

Linear observation model, ``x ↦ Hx``, or ``x ↦ xH^*``.

```julia
H = randn(2,3)
lobs = LinearObserver(H) # standard linear observer for vectors of size 3
x = randn(3)
lobs(x) == H*x # true
```

A positive `depth` parameter indicates that the action is
on matrices instead of vectors.

```julia
H = randn(2,3)
lobs = LinearObserver(H; depth=4) # standard linear observer for matrices of size 3x4
x = randn(3,4)
lobs(x) == H*x # true
```

There is a right version as well:
```julia
H = randn(2,3)
lobs = LinearObserver(H, RightAction(); depth=4)
x = randn(4,3)
lobs(x) == x*H' # true
```
"""
LinearObserver(matrix, conv=LeftAction(); depth=0) = LinearObserver{typeof(conv), typeof(matrix), typeof(depth)}(matrix, depth)




Base.show(io::IO, obs::LinearObserver{TAD}) where {TAD} = begin
    depth_string = obs.depth > 0 ? "; depth=$(obs.depth)" : ""
    print(io, "LinearObserver($(obs.matrix), $(TAD)()" * depth_string * ")")
end

true_adjoint_🙄(v::AbstractVector) = conj(v)
true_adjoint_🙄(M::AbstractMatrix) = M'

get_measurement(obs::LinearObserver{LeftAction}, x) = obs.matrix * x
get_measurement(obs::LinearObserver{RightAction}, x) = true_adjoint_🙄(obs.matrix * true_adjoint_🙄(x))

get_tan_observer(obs::LinearObserver, A::AbstractGroupAction{LeftAction}, x::Any, ::Any) = ξ -> get_measurement(obs, apply_diff_group(A, Identity(base_group(A)), ξ, x))

_make_depth_tuple(obs::LinearObserver, i) = let dim = size(obs.matrix, i), depth = obs.depth
        return depth == 0 ? dim : (dim, depth)
    end

_reverse(t::Tuple) = reverse(t)
_reverse(i::Integer) = i

codomain_tuple(obs::LinearObserver{LeftAction}) = _make_depth_tuple(obs, 1)
codomain_tuple(obs::LinearObserver{RightAction}) = _reverse(_make_depth_tuple(obs, 1))

domain_tuple(obs::LinearObserver{LeftAction}) = _make_depth_tuple(obs, 2)
domain_tuple(obs::LinearObserver{RightAction}) = _reverse(_make_depth_tuple(obs, 2))

observation_space(obs::LinearObserver) = Euclidean(codomain_tuple(obs)...)
observed_space(obs::LinearObserver) = Euclidean(domain_tuple(obs)...)
