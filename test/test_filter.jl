using Test
using GeometricFilter
using Manifolds

using SparseArrays
using PDMats
using LinearAlgebra


import Random

rng = Random.default_rng()


# TODO: test a motion on a sphere 

# TODO
# 1) set up a regular Kalman filter (with flat spaces)
# 2) compare with standard implementation of Kalman filter

@testset "Degenerate Process Noise" begin
    dim = 3
    size = 2
    G = MultiDisplacementGroup(dim,size)
    A =DualGroupOperationAction(G)
    pdiag = spzeros(manifold_dimension(G))
    idx = first(axes(pdiag))
    pdiag[MultiAffine.normal_indices(G, idx; pos=1)] .= 1.
    pdiag[MultiAffine.factor_indices(G, idx)] .= 2.
    cov = PDiagMat(pdiag)
    ccov = PDMatsSingular.Covariance(PDiagMat(pdiag))
    dist = ActionDistribution(DualGroupOperationAction(G), identity_element(G), cov)
    pnoise = action_noise(dist)
    cdist = ActionDistribution(DualGroupOperationAction(G), identity_element(G), ccov)
    cpnoise = action_noise(cdist)
    run_simulation(d, n) = simulate_filter(d, [StochasticMotion(ZeroMotion(A), n)], [Observation()], PositionPerturbation(rng))
    @test_throws PosDefException run_simulation(dist, pnoise)
    @test_throws PosDefException run_simulation(cdist, pnoise)
    @test_throws PosDefException run_simulation(dist, cpnoise)
    run_simulation(cdist, cpnoise)
end

@testset "Test Flat Filter Partial Observation" begin
	  lin = [0 1.;0 0]
    trans = zeros(2)
    motion = GeometricFilter.FlatAffineMotion(lin, trans)
    A = AffineMotions.get_action(motion)
    x0 = [0.0, 20]
    # pnoise = ActionNoise(A, x->[1. 0;0 3.], DefaultOrthonormalBasis())
    # pnoise = ActionNoise(A, [1. 0;0 3.], DefaultOrthonormalBasis())
    # pnoise = ActionNoise(A, x->PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
    pnoise = ActionNoise(A, PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
    observer = LinearObserver([1. 0])
    onoise = IsotropicNoise(observation_space(observer), sqrt(10.0))

    # D0 = ActionDistribution(A, x0, 5*PDMats.PDiagMat([1.,1.]), DefaultOrthonormalBasis())
    D0 = ActionDistribution(A, x0, PDMats.ScalMat(2, 5.0), DefaultOrthonormalBasis())
    D0_ = ActionDistribution(A, x0, 5)
    D0__ = ActionDistribution(A, x0, 5, DefaultOrthonormalBasis())
    dt = .02
    predict(D0, StochasticMotion(dt*motion, pnoise))
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
    A = AffineMotions.get_action(motion)
    V = group_manifold(A)
    G = base_group(A)
    x0 = zeros(1)
    # pnoise = IsotropicNoise(V, 2.)
    pnoise = ActionNoise(A, 2.0^2)
    onoise = IsotropicNoise(V, 1.)
    observer = LinearObserver([1.;;])
    x1 = pnoise(rng, integrate(motion, x0))
    # @show x1
    y1 = observer(x1)
    z1 = onoise(rng, y1)
    # @show z1

    D0 = ActionDistribution(A, x0, PDMats.ScalMat(1,1.), DefaultOrthonormalBasis())
    D1 = predict(D0, StochasticMotion(motion, pnoise))
    # @code_warntype predict(D0, StochasticMotion(motion, pnoise))
    D1_ = update(D1, Observation(observer, onoise, z1))
    # @show D1_
end


@testset "Test Simulation" begin
    motion = FlatAffineMotion([0. 1.;0 0 ], [0.,0])
    pnoise = ActionNoise(AffineMotions.get_action(motion), 1.)
    observer = LinearObserver([1. 0])
    onoise = IsotropicNoise(observation_space(observer), 1.)
    D0 = ActionDistribution(AffineMotions.get_action(motion), [0., 0], 1.)
    T = 10
    signal_ = generate_signal(fill(motion, T), D0.μ)
    @testset "Signal" begin
        signal = accumulate(1:T; init=D0.μ) do x, i
            return integrate(motion, x)
        end
        @test signal == signal_[begin+1:end]
    end
    selector(i, N) = (i % N) == 0
    observations = [selector(i,T÷2) ? Observation(observer, onoise, onoise(rng, observer(s))) : EmptyObservation() for (i,s) in enumerate(signal_)]
    sms = fill(StochasticMotion(motion, pnoise), T)
    @testset "Simulation Mode $mode" for mode in [PositionPerturbation, SensorPerturbation]
        nbs, dists = simulate_filter(D0, sms, observations, mode(rng))
        length(dists) == T
        @test length(unique(nbs)) == 3
    end
    @testset "Wrong nb of observations" begin
        nbs, dists = @test_logs (:info,) simulate_filter(D0, sms, observations[begin:end-2], DataMode())
        @test all(isassigned(dists, i) for i in eachindex(dists))
    end
    @testset "Wrong nb of motions" begin
        k = 2
        nbs, dists = @test_logs (:warn,) simulate_filter(D0, sms[begin:end-k], observations, DataMode())
        @test length(dists) == length(sms) - k
    end

end

@testset "Test Filter" begin
    rng = Random.default_rng(3)
    dim = 2
    G = MultiDisplacementGroup(dim, 1)
    action = GroupOperationAction(G)
    # vel = rand(rng, GeometricFilter.algebra(G))
    vel = zero_vector(G, identity_element(G))
    vel[firstindex(vel)] = 1.
    rm = RigidMotion(action, vel)

    x0 = identity_element(G)
    x1 = integrate(rm, x0)
    # @show x1

    T = 10
    process_noise = ActionNoise(action, 0.5)
    stoch_motion = StochasticMotion(rm, process_noise)
    @test_throws MethodError predict(ActionDistribution(DualGroupOperationAction(G), x0, 1.), stoch_motion)
    sms = fill(stoch_motion, T)
    rng = Random.default_rng()
    signal = generate_signal(sms, x0, PositionPerturbation(rng))
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
    H = GU.get_op_matrix(G, GeometricFilter.plain_manifold(observation_space(observer)), measure, tobs, DefaultOrthonormalBasis(), DefaultOrthonormalBasis())
    # @show H
    # what should H look like?
    ob_noise = IsotropicNoise(Mobs, x -> .1)
    observations = [observer(s) for s in signal]
    pobs = [ob_noise(rng, ob) for ob in observations]

    # start new test set here?

    # istate = ActionDistribution(action, x0, initial_uncertainty, B)
    istate = ActionDistribution(action, x0, 1.0)
    clean = predict(istate, rm)
    # process_noise = IsotropicNoise(G, x->.1)
    process_noise = ActionNoise(GroupOperationAction(G), .1)


    obnoise = IsotropicNoise(group_manifold(A), x -> .1)

    # Σ_, pred, H, G = GeometricFilter.prepare_correction(clean, observer, obnoise)
    Σ_, G = GeometricFilter.prepare_correction(clean, H, obnoise, measure)


    update(clean, Observation(observer, obnoise, measure))

end

@testset "Test Translation" begin
    G = MultiDisplacementGroup(2)
    A = GroupOperationAction(G)
    ξ = rand(rng, TangentSpace(G, identity_element(G)))
    m = TranslationMotion(G, ξ, LeftSide())
    D = ActionDistribution(A, identity_element(G), 1.)
    # pnoise = IsotropicNoise(G, x->1.)
    pnoise = ActionNoise(GroupOperationAction(G), 1.)
    predict(D, StochasticMotion(m, pnoise))
end

@testset "Simulation" begin
    m = FlatAffineMotion([1. 0;0 1], zeros(2))
    p = zeros(2)
    T = 10
    signal = generate_signal(fill(m,T),p)
    @test length(signal) == T+1
end
