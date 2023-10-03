using Test
using GeometricFilter
using Manifolds

using SparseArrays
using DataFrames
using PDMats
using LinearAlgebra


import Random

rng = Random.default_rng()


# TODO: test a motion on a sphere 

# TODO
# 1) set up a regular Kalman filter (with flat spaces)
# 2) compare with standard implementation of Kalman filter

@testset "Test Flat Filter Partial Observation" begin
	  lin = [0 1.;0 0]
    trans = zeros(2)
    motion = GeometricFilter.FlatAffineMotion(lin, trans)
    A = get_action(motion)
    x0 = [0.0, 20]
    # pnoise = ActionNoise(A, x->[1. 0;0 3.], DefaultOrthonormalBasis())
    # pnoise = ActionNoise(A, [1. 0;0 3.], DefaultOrthonormalBasis())
    # pnoise = ActionNoise(A, x->PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
    pnoise = ActionNoise(A, PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
    observer = LinearObserver([1. 0])
    onoise = IsotropicNoise(observation_space(observer), sqrt(10.0))

    # D0 = ProjLogNormal(A, x0, 5*PDMats.PDiagMat([1.,1.]), DefaultOrthonormalBasis())
    D0 = ProjLogNormal(A, x0, PDMats.ScalMat(2, 5.0), DefaultOrthonormalBasis())
    D0_ = ProjLogNormal(A, x0, 5)
    D0__ = ProjLogNormal(A, x0, 5, DefaultOrthonormalBasis())
    dt = .02
    predict(D0, dt*motion, pnoise)
    update(D0, Observation(observer, onoise, [1.]))
    update(D0, EmptyObservation())
    update(D0, Observation())
end

@testset "Test Flat Filter" begin
    rng = Random.default_rng(3)
    lin = zeros(2,2)
    trans = [1., 0]
    m = FlatAffineMotion(lin, trans)
    x0 = zeros(2)
    x1 = integrate(m, x0)
    obs = LinearObserver([1 0])
    measure = obs(x1)
    @test measure ≈ [1.0]
end

# TODO: such a common task: make a helper function?
@testset "Test scalar flat filter" begin
    rng = Random.default_rng(3)
    # same test system as in https://github.com/mschauer/Kalman.jl/blob/master/test/testsystem.jl
    lin = [0.5;;]
    trans = [0.]
    motion = GeometricFilter.FlatAffineMotion(lin, trans)
    A = get_action(motion)
    V = group_manifold(A)
    G = base_group(A)
    x0 = zeros(1)
    # pnoise = IsotropicNoise(V, 2.)
    pnoise = ActionNoise(GroupOperationAction(G), 2.0^2)
    onoise = IsotropicNoise(V, 1.)
    observer = LinearObserver([1.;;])
    x1 = pnoise(rng, integrate(motion, x0))
    # @show x1
    y1 = observer(x1)
    z1 = onoise(rng, y1)
    # @show z1

    D0 = ProjLogNormal(A, x0, PDMats.ScalMat(1,1.), DefaultOrthonormalBasis())
    D1 = predict(D0, motion, pnoise)
    D1_ = update(D1, Observation(observer, onoise, z1))
    # @show D1_
end


@testset "Test Simulation" begin
    motion = FlatAffineMotion([0. 1.;0 0 ], [0.,0])
    pnoise = ActionNoise(get_action(motion), 1.)
    observer = LinearObserver([1. 0])
    onoise = IsotropicNoise(observation_space(observer), 1.)
    D0 = ProjLogNormal(get_action(motion), [0., 0], 1.)
    T = 10
    selector(i, N) = (i % N) == 0
    signal = accumulate(1:T; init=D0.μ) do x, i
        return integrate(motion, x)
    end
    signal_ = generate_signal(fill(motion, T), D0.μ)
    @test signal == signal_
    observations = [selector(i,T÷2) ? Observation(observer, onoise, onoise(rng, observer(s))) : EmptyObservation() for (i,s) in enumerate(signal)]
    sms = fill(StochasticMotion(motion, pnoise), T)
    res = simulate_filter(rng, D0, sms, observations; streak_nb=:nb)
    @test res isa DataFrame
    @test names(res) == ["nb", "dist"]
    @test nrow(res) == T
    @test length(groupby(res, :nb)) == 3
end

@testset "Test Filter" begin
    rng = Random.default_rng(3)
    dim = 2
    G = MultiDisplacement(dim, 1)
    action = GroupOperationAction(G)
    # vel = rand(G; vector_at=identity_element(G))
    vel = zero_vector(G, identity_element(G))
    vel[firstindex(vel)] = 1.
    rm = RigidMotion(action, vel)

    x0 = identity_element(G)
    x1 = integrate(rm, x0)
    # @show x1

    T = 10
    process_noise = ActionNoise(action, 0.5)
    stoch_motion = StochasticMotion(rm, process_noise)
    @test_throws MethodError predict(ProjLogNormal(DualGroupOperationAction(G), x0, 1.), stoch_motion)
    sms = fill(stoch_motion, T)
    rng = Random.default_rng()
    signal = generate_signal(rng, sms, x0)
    # @show signal
    # @show length(signal)

    x = last(signal)
    action = GroupOperationAction(G)
    ref = identity_element(G)

    A = MultiAffineAction(G)
    observer = PositionObserver(A)
    measure = observer(signal[3])
    tobs = GeometricFilter.get_tan_observer(observer, action, x, measure)
    Mobs = observation_space(observer)
    B = get_basis(G, identity_element(G), DefaultOrthonormalBasis())
    H = GeometricFilter.get_op_matrix(G, observation_space(observer), measure, tobs, DefaultOrthonormalBasis(), DefaultOrthonormalBasis())
    # @show H
    # what should H look like?
    ob_noise = IsotropicNoise(Mobs, x -> .1)
    observations = [observer(s) for s in signal]
    pobs = [ob_noise(rng, ob) for ob in observations]

    # start new test set here?

    # istate = ProjLogNormal(action, x0, initial_uncertainty, B)
    istate = ProjLogNormal(action, x0, 1.0)
    clean = predict(istate, rm)
    # process_noise = IsotropicNoise(G, x->.1)
    process_noise = ActionNoise(GroupOperationAction(G), .1)
    # noisy = GeometricFilter.add_process_noise(clean, process_noise)


    obnoise = IsotropicNoise(group_manifold(A), x -> .1)

    # Σ_, pred, H, G = GeometricFilter.prepare_correction(clean, observer, obnoise)
    Σ_, G = GeometricFilter.prepare_correction(clean, H, obnoise, measure)


    update(clean, Observation(observer, obnoise, measure))

end

@testset "Test Translation" begin
    G = MultiDisplacement(2)
    A = GroupOperationAction(G)
    ξ = rand(rng, G; vector_at=Identity(G))
    m = TranslationMotion(G, ξ)
    D = ProjLogNormal(A, identity_element(G), 1.)
    # pnoise = IsotropicNoise(G, x->1.)
    pnoise = ActionNoise(GroupOperationAction(G), 1.)
    predict(D, m, pnoise)
end



