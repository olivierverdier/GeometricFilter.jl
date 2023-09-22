using GeometricFilter
using Manifolds
using Infiltrator

τ = 2*π

G = MultiDisplacement(2, 2)

m_ma = MultiAffineMotion(G, [0 0; 1.0 0], LeftAction())

