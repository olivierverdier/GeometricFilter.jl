
"""
    update(prior::AbstractProjLogNormal, observation::AbstractObservation)

Update the uncertainty (the prior) given an observation.
Typically, it is used as
```julia
update(prior, Observation(observer, noise, measurement))
```
 with a given observer, an associated noise, and a measurement
on the observation manifold (the manifold of the `observer`).
"""
update(prior::AbstractProjLogNormal, ob::Observation) = _update(prior, ob.observer, ob.noise, ob.measurement)

@deprecate update(
    prior::AbstractProjLogNormal,
    observer, # Observer
    noise, # Observation noise
    measurement, # Actual measurement
    )  update(prior, Observation(observer, noise, measurement))

function _update(
    prior::AbstractProjLogNormal,
    observer, # Observer
    noise, # Observation noise
    measurement, # Actual measurement
    )
    x = Distributions.mean(prior)
    pred = observer(x)

    obs_basis = get_basis_at(noise, pred)

    H = get_obs_matrix(prior, observer, pred, obs_basis)
    Σ_, gain = prepare_correction(prior, H, noise, pred)
    N = observation_space(observer)
    B = get_lie_basis(prior)
    action = get_action(prior)

    innovation = log(N, pred, measurement)
    obs_basis = get_basis_at(noise, pred)
    innovec = get_coordinates(N, pred, innovation, obs_basis)
    x_ = point_correction(x, action, gain, B, innovec)
    return update_mean_cov(prior, x_, Σ_)
end



update(prior, ::EmptyObservation) = prior


#--------------------------------
# Update Helpers
#--------------------------------

function get_obs_matrix(prior, observer, pred, obs_basis)
    x = Distributions.mean(prior)
    action = get_action(prior)
    G = base_group(action)

    obs_op = get_tan_observer(observer, action, x, pred)

    basis = get_lie_basis(prior)
    H = GroupTools.get_op_matrix(G, observation_space(observer), pred, obs_op, basis, obs_basis)
    return H
end

function prepare_correction(
    prior::AbstractProjLogNormal, # prior from forecasting
    obs_matrix,
    noise, # observation noise
    pred,
)
    Σ = Distributions.cov(prior)
    Σy = get_covariance_at(noise, pred)

    H = obs_matrix

    # obs_cov = H*Σ*H' + Σy
    obs_cov = PDMats.X_A_Xt(Σ, H) + Σy

    Gain = Σ*H'*inv(obs_cov)

    A = LinearAlgebra.I - Gain*H

    # Σ_ = A*Σ*A' + Gain*Σy*Gain'
    Σ_ = PDMats.AbstractPDMat(PDMats.X_A_Xt(Σ,A) + PDMats.X_A_Xt(Σy, Gain))

    return Σ_, Gain
end

# TODO: compute observation and tangent obs at the same time?

function point_correction(
    x,
    action,
    # just compute the innovation?
    gain,
    B::AbstractBasis,
    innovation_coords, # coordinates of point at tangent space of prediction
    )
    group = base_group(action)
    increment_coord = gain*innovation_coords
    increment = get_vector_lie(group, increment_coord, B)
    movement = exp_lie(group, increment)
    x_ = apply(action, movement, x)
    return x_
end



