abstract type FilteringMode end

struct DataMode <: FilteringMode end

struct SimulationMode{T,TRNG} <: FilteringMode
    rng::TRNG
end

abstract type PerturbationType  end
struct SensorPerturbationMode <: PerturbationType end
struct PositionPerturbationMode <: PerturbationType end

SimulationMode(rng::Random.AbstractRNG, t::PerturbationType) = SimulationMode{typeof(t),typeof(rng)}(rng)

PositionPerturbation(rng) = SimulationMode(rng, PositionPerturbationMode())
SensorPerturbation(rng) = SimulationMode(rng, SensorPerturbationMode())

"""
    generate_signal(motions::AbstractVector{<:AbstractMotion}, x0)

Integrate the sequence of motions, starting at the point ``x_0``.
"""
generate_signal(
    motions::AbstractVector{<:AbstractMotion},
    x0 # initial point
) =
    accumulate(motions; init=x0) do x, m
        return integrate(m, x)
    end

"""
    generate_signal(rng::AbstractRNG,
      motions::AbstractVector{<:AbstractStochasticMotion}, x0)

Integrate one sample path the sequence of stochastic motions, starting at the point ``x_0``.
"""
function generate_signal(
    rng::Random.AbstractRNG,
    stoch_motions::AbstractVector{<:AbstractStochasticMotion},
    x0 # initial point
    )
    res = accumulate(stoch_motions; init=x0) do x, sm
        return integrate(rng, sm, x)
    end
    return res
end




function get_noisy_observations(
    rng,
    observations, # vector of observations
    noises, # vector of observation noises
    )
    return [noise(rng, obs) for (noise, obs) in zip(noises, observations)]
end

_is_obs(::AbstractObservation) = true
_is_obs(::EmptyObservation) = false

prediction_step(::DataMode, D, sm) = predict(D, sm)
function prediction_step(fm::SimulationMode{SensorPerturbationMode}, D, sm)
    x = Distributions.mean(D)
    sm_ = sensor_perturbation(fm.rng, sm, x)
    return prediction_step(DataMode(), D, sm_)
end
function prediction_step(fm::SimulationMode{PositionPerturbationMode}, D, sm)
    D_ = prediction_step(DataMode(), D, sm)
    return apply_noise(fm.rng, sm, D_)
end

"""
    simulate_filter(::FilteringMode,
        ::ProjLogNormal,
        ::Vector{<:StochasticMotion},
        ::Vector{<:Observation})
          :: DataFrame

Runs a simulation of the filter, given a starting distribution,
 a vector of stochastic motions, and a vector of observations.
Return a dataframe with a column containing the “streak number”,
and one column containing the filtered distributions.
There are three main mode to use as `FilterMode`:
- `DataMode`: when it is used with real data
- `SensorPerturbation(rng)` for simulated perturbation of sensor inputs
- `PositionPerturbation(rng)` for simulated perturbations of positions
"""
function simulate_filter(fm::FilteringMode, D0::AbstractProjLogNormal, stochastic_motions, observations; streak_nb=:streak_nb, dist=:dist)
    res = accumulate(zip(stochastic_motions, observations); init=(1=>D0)) do (k, D), (sm, obs)
        new_k = _is_obs(obs) ? k+1 : k

        D__ = prediction_step(fm, D, sm)

        D___ = update(D__, obs)
        return (new_k => D___)
    end
    res_ = map(res) do (i,d)
        return NamedTuple{(streak_nb, dist)}([i, d])
    end
    df = DataFrames.DataFrame(res_)
    return df
end
