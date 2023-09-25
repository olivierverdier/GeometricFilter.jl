
"""
    LinearObserver(H::AbstractMatrix)

Linear observation model, ``x ↦ Hx``.
"""
struct LinearObserver{TV} <: AbstractObserver
    matrix::TV
end

Base.show(io::IO, obs::LinearObserver) = print(io, "LinearObserver($(obs.matrix))")

get_measurement(obs::LinearObserver, x) = obs.matrix * x
get_tan_observer(obs::LinearObserver, ::AbstractGroupAction{LeftAction}, x, y) = ξ -> obs.matrix * ξ
observation_space(obs::LinearObserver) = Euclidean(size(obs.matrix, 1))
