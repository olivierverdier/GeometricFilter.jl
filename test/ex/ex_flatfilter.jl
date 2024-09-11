using GeometricFilter
using Manifolds

import PDMats
import Random
rng = Random.default_rng()

lin = [0 1.0; 0 0]
trans = zeros(size(lin,1))
motion = FlatAffineMotion(lin, trans)

x0 = Float64[0, 20]
pnoise = ActionNoise(AffineMotions.get_action(motion),  PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
observer = LinearObserver([1.0 0])
onoise = IsotropicNoise(observation_space(observer), sqrt(10.0))

D0 = ActionDistribution(x0, update_cov(pnoise, PDMats.ScalMat(2, 5.)))

data = randn(rng, 140)
dt = 0.02

res = accumulate(data; init=D0) do D, z
    D_ = update(D, Observation(observer, onoise, [z]))
    D__ = predict(D_, StochasticMotion(dt*motion, pnoise))
    return D__
end
