
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
    Σ_, pred, H, G = prepare_correction(prior, observer, noise)
    x = Distributions.mean(prior)
    N = get_manifold(observer)
    B = prior.B
    action = get_action(prior)

    innovation = log(N, pred, measurement)
    obs_basis = get_basis_at(noise, pred)
    innovec = get_coordinates(N, pred, innovation, obs_basis)
    x_ = point_correction(x, action, G, B, innovec)
    return update_mean_cov(prior, x_, Σ_)
end



update(prior, ::EmptyObservation) = prior


#--------------------------------
# Update Helpers
#--------------------------------

function prepare_correction(
    prior::AbstractProjLogNormal, # prior from forecasting
    observer, # Observer
    noise, # observation noise
    )
    x = Distributions.mean(prior)
    pred = observer(x)

    action = get_action(prior)
    G = base_group(action)

    obs_op = get_tan_observer(observer, action, x, pred)
    # TODO: create a simpler get_obs_matrix tailored for observations?
    basis = prior.B
    obs_basis = get_basis_at(noise, pred)
    H = get_op_matrix(G, get_manifold(observer), pred, obs_op, basis, obs_basis)

    Σ = Distributions.cov(prior)
    Σy = get_covariance_at(noise, pred, obs_basis)

    # obs_cov = H*Σ*H' + Σy
    obs_cov = PDMats.X_A_Xt(Σ, H) + Σy

    G = Σ*H'*inv(obs_cov)

    A = LinearAlgebra.I - G*H

    # Σ_ = A*Σ*A' + G*Σy*G'
    Σ_ = PDMats.PDMat(PDMats.X_A_Xt(Σ,A) + PDMats.X_A_Xt(Σy, G))

    return Σ_, pred, H, G
end

# TODO: compute observation and tangent obs at the same time?

function point_correction(
    x,
    action,
    # just compute the innovation?
    gain,
    B::AbstractBasis,
    innovation, # point at tangent space of prediction
    )
    group = base_group(action)
    increment_coord = gain*innovation
    increment = get_vector_lie(group, increment_coord, B)
    movement = exp_lie(group, increment)
    x_ = apply(action, movement, x)
    return x_
end



