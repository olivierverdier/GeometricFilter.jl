
"""
A model of a map from a manifold to a covariance on a Lie algebra.
"""
abstract type ProcessNoise{TA<:AbstractGroupAction{LeftAction}} <: AbstractNoise end

const AbstractActionNoise{TA} = ProcessNoise{TA}
#--------------------------------
# AbstractActionNoise Interface
#--------------------------------
"""
    get_lie_covariance_at(
      ::AbstractActionNoise,
      x, # point in manifold M
      B, # AbstractBasis of Alg(G)
      ) :: AbstractPDMat

Covariance on Alg(G), where G is the action group.
"""
function get_lie_covariance_at end

"""
    get_group(::AbstractActionNoise) :: AbstractDecoratedManifold

The group on which this noise model is defined.
"""
function get_group end
#--------------------------------

"""
    ActionNoise(
        action, # group action G ⊂ Diff(M)
        covariance, # function M -> AbstractPDMat
        basis # fixed basis of Alg(G)
    )

Covariance defined on Lie algebra.
The corresponding distribution centred at x₀ 
is the push forward of the normal distribution on the Lie
algebra by the function ξ ↦ exp(ξ)⋅x₀.
"""
struct ActionNoise{TA,TF<:Function,TB}  <: AbstractActionNoise{TA}
    action::TA # group action G ⊂ Diff(M)
    covariance::TF # function M -> AbstractPDMat
    basis::TB # fixed basis of Alg(G)
end

Base.show(io::IO, n::ActionNoise) = print(io, "ActionNoise($(n.action), $(n.covariance), $(n.basis))")


get_group(a::ActionNoise)  = base_group(a.action)
sample_space(a::ActionNoise)  = group_manifold(a.action)

"""
    ActionNoise(A::GroupAction, Σ::PDMat)

Convenience method to create an action noise with a constant covariance Σ, by default with respect to the standard metric of the Lie algebra.
"""
ActionNoise(
    A::AbstractGroupAction,
    Σ::PDMats.AbstractPDMat,
    B=DefaultOrthonormalBasis()) = ActionNoise(A, ConstantFunction(Σ), B)

"""
    ActionNoise(A::GroupAction, σ::Number)

Convenience method to create an action noise with a constant, isotropic covariance, by default with respect to the standard metric of the Lie algebra.
"""
function ActionNoise(
    A::AbstractGroupAction,
    σ::Number=1.0,
    B=DefaultOrthonormalBasis())
    G = base_group(A)
    dim = manifold_dimension(G)
    return ActionNoise(A, PDMats.ScalMat(dim, σ), B)
end

rescale_noise(n::ActionNoise, scale) = ActionNoise(n.action, x -> scale^2*n.covariance(x), n.basis)

rescale_noise(n::ActionNoise{<:Any,<:ConstantFunction}, scale) = ActionNoise(n.action, scale^2*n.covariance, n.basis)

"""
    update_cov(n::ActionNoise, Σ)

New action noise with same action and basis but with new (possibly point dependent) covariance Σ.
"""
update_cov(n::ActionNoise, Σ) = ActionNoise(n.action, Σ, n.basis)

_basis_error_message(B1, B2) = "Changing from basis\n\t$B1\nto\n\t$B2\nis not implemented"

function get_lie_covariance_at(
    noise::ActionNoise, # noise for action(M,G)
    x, # on M
    B::AbstractBasis, # basis of Alg(G)
    ) 
    if !is_point(sample_space(noise), x)
        throw(ErrorException("x should be a point on the manifold"))
    end
    if B != noise.basis
        throw(ErrorException(_basis_error_message(B, noise.basis)))
    end
    return noise.covariance(x)
end

get_lie_covariance_at(noise::ActionNoise, x) = get_lie_covariance_at(noise, x, noise.basis)

function get_covariance_at(
    noise::ActionNoise,
    x, # point on Manifold M
    B # basis of T_xM
    ) 
    A = noise.action
    Σ = noise.covariance(x)
    BG = noise.basis
    P = GU.get_proj_matrix(A, x, BG, B)
    return PDMats.AbstractPDMat(PDMats.X_A_Xt(Σ, P))
end

function get_covariance_at(
    noise::ActionNoise{TA},
    ::Identity,
    B
    ) where {TA<:GroupOperationAction}
    if B != noise.basis
        throw(ErrorException(_basis_error_message(B, noise.basis)))
    end
    G = base_group(noise.action)
    return noise.covariance(Identity(G))
end

function add_noise(
    noise::ActionNoise,
    rng::Random.AbstractRNG,
    point)
    cov = noise.covariance(point)
    dist = ProjLogNormal(noise.action, point, cov, noise.basis)
    return rand(rng, dist)
end

add_noise(noise::ActionNoise{<:Any, <:ConstantFunction}, rng::Random.AbstractRNG, point) = rand(rng, ProjLogNormal(point, noise))


"""
Obsolete: in this basis, the covariance matrix is the identity.
"""
function get_adapted_basis_at(noise::ActionNoise, x)
    A = noise.action
    BM = ManifoldsBase.get_basis(group_manifold(A), x, DefaultOrthonormalBasis())
    L = GU.get_proj_matrix(A, x, noise.basis, BM)
    res = LinearAlgebra.svd(L)
    vec_mat = res.U .* res.S'
    return CachedBasis(DefaultOrthonormalBasis(), collect(eachcol(vec_mat)))
end
