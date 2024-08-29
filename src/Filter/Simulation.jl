function generate_signal_from(f, dyns, x0)
    N = length(dyns)
    result = Vector{typeof(x0)}(undef, N+1)
    result[firstindex(result)] = x0
    # indices = Iterators.drop(first(axes(result)), 1)
    indices = first(axes(result))
    last_x = foldl(zip(indices, dyns); init=x0) do x, (i,d)
        result[i] = x
        x_ = f(x, d)
        return x_
    end
    result[end] = last_x
    return result
end

"""
    generate_signal(motions::AbstractVector{<:AbstractMotion}, x0)

Integrate the sequence of motions, starting at the point ``x_0``.
"""
generate_signal(
    motions::AbstractVector{<:AbstractMotion},
    x0 # initial point
) = generate_signal_from(motions, x0) do x, m
    return integrate(m, x)
end

"""
    generate_signal(
      motions::AbstractVector{<:AbstractStochasticMotion}, x0,
      ::FilteringMode
    )

Integrate one sample path the sequence of stochastic motions, starting at the point ``x0``.
"""
generate_signal(
    stoch_motions::AbstractVector{<:AbstractStochasticMotion},
    x0, # initial point
    fm::FilteringMode,
) = generate_signal_from(stoch_motions, x0) do x, sm
        return simulate(sm, x, fm)
    end



function get_noisy_observations(
    rng,
    observations, # vector of observations
    noises, # vector of observation noises
    )
    return [noise(rng, obs) for (noise, obs) in zip(noises, observations)]
end

_increase_if_observation(::AbstractObservation, k) = k+1
_increase_if_observation(::EmptyObservation, k) = k


"""
    simulate_filter(
        ::ProjLogNormal,
        ::Vector{<:StochasticMotion},
        ::Vector{<:Observation},
        ::FilteringMode,
          ) :: Tuple{Vector(Int), Vector(ProjLogNormal)}

Runs a simulation of the filter, given a starting distribution,
 a vector of stochastic motions, and a vector of observations.
Return a dataframe with a column containing the “streak number”,
and one column containing the filtered distributions.
There are three main mode to use as `FilterMode`:
- `DataMode`: when it is used with real data
- `SensorPerturbation(rng)` for simulated perturbation of sensor inputs
- `PositionPerturbation(rng)` for simulated perturbations of positions
"""
simulate_filter(D0::AbstractProjLogNormal, stochastic_motions, observations, fm::FilteringMode) =
    simulate_filter_from(D0, stochastic_motions, observations) do D, sm
    return prediction_step(fm, D, sm)
end

"""
    simulate_filter(prediction_step,
    D0,
    stochastic_motions,
    observations)

Run a filter from stochastic motions and observations.
The argument `prediction_step` can be used to
modify the standard behaviour of `predict`,
for instance by adding sensor noise, position noise,
or both.
"""
function simulate_filter_from(prediction_step, D0::AbstractProjLogNormal, stochastic_motions, observations)
    N = length(stochastic_motions)
    nb_missing_obs = N - length(observations)
    if nb_missing_obs > 0
        @info "Missing Observations"
        observations_ = [observations; fill(Observation(), nb_missing_obs)]
    else
        if nb_missing_obs < -1
            @warn "$(-nb_missing_obs) Movements Missing"
        end
        observations_ = observations
    end
    # dists = Vector{typeof(D0)}(undef, N)
    dists = Vector{AbstractProjLogNormal}(undef, N)
    obs_count = Vector{Int}(undef, N)
    foldl(zip(axes(obs_count, 1), axes(dists, 1), stochastic_motions, observations_); init=(0=>D0)) do (k, D), (ki, di, sm, obs)
        new_k = _increase_if_observation(obs, k)
        D_ = update(D, obs)

        obs_count[ki] = new_k
        dists[di] = D_

        D__ = prediction_step(D_, sm)

        return (new_k => D__)
    end
    return obs_count, dists
end

prediction_step(::DataMode, D, sm) = predict(D, sm)
function prediction_step(fm::SimulationMode{SensorPerturbationMode}, D, sm)
    x = Distributions.mean(D)
    sm_ = sensor_perturbation(fm.rng, sm, x)
    return prediction_step(DataMode(), D, sm_)
end
function prediction_step(fm::SimulationMode{PositionPerturbationMode}, D, sm)
    D_ = prediction_step(DataMode(), D, sm)
    return update_mean(D_, get_noise(sm)(fm.rng, Distributions.mean(D_)))
end

