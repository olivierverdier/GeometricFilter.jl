include("./ex_localisation.jl")

import Random
rng = Random.default_rng()

full_path = straight_path(1., 0.2, 30)
poses = compute_poses(full_path)
vels = compute_velocities(G, poses)
motions = compute_motions(A, vels)
smotions = map(m -> StochasticMotion(m, ActionNoise(A, 0.01), PositionMode()), motions)
# smotions = map(m -> StochasticMotion(m, IsotropicNoise(G, 0.01), PositionMode()), motions)
init_pose = identity_element(G)

poses = generate_signal(motions, init_pose)

noisy_trajectory = accumulate(smotions; init=init_pose) do pose, sm
    return integrate(rng, sm, pose)
end


get_positions(poses) = hcat([submanifold_component(p,1) for p in poses]...)

using Plots

function plot_positions(pose_streaks, plt=plot(); kwargs...)
	# plot = plot_landmarks(plt)
	plt = plot()
	for (i, streak) in enumerate(pose_streaks)
		#colors = [Colors.Gray(x) for x in LinRange(1,0.,size(positions,2))]
		positions = get_positions(streak)
		scatter!(plt, positions[1,:], positions[2,:], #markercolor=colors,
		markersize=2, markerstrokewidth=.1, label="$i"; kwargs...)
	end
	return plt
end
