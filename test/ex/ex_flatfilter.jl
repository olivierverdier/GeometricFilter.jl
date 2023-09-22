using GeometricFilter
using Manifolds

import PDMats
import Random
rng = Random.default_rng()

dim = 2
lin = [0 1.0; 0 0]
trans = zeros(dim)
motion = FlatAffineMotion(lin, trans)

x0 = Float64[0, 20]
pnoise = ActionNoise(get_action(motion),  PDMats.PDiagMat([1.,3.]), DefaultOrthonormalBasis())
observer = LinearObserver([1.0 0])
onoise = IsotropicNoise(get_manifold(observer), sqrt(10.0))

D0 = ProjLogNormal(get_action(motion), x0, PDMats.ScalMat(2, 5.), DefaultOrthonormalBasis())

data = randn(rng, 140)
dt = 0.02

res = accumulate(data; init=D0) do D, z
    D_ = update(D, observer, onoise, [z])
    D__ = predict(D_, dt*motion, pnoise)
    return D__
end
