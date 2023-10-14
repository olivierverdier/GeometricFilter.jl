using GeometricFilter
using Manifolds

τ = 2*π

G = MultiDisplacement(2, 2)

m_ma = AdjointLinearMotion(G, [0 1; 0 0], LeftAction())

