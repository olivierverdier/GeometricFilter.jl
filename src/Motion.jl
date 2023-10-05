@doc raw"""
General interface for maps of the form ``φ \colon M \to \mathfrak{g}``,
where ``M`` is a manifold, and ``\mathfrak{g}`` is the Lie algebra
of a Lie group ``G`` action on the manifold ``M``.

```julia
G = SpecialOrthogonal(3)
ξ = rand(G; vector_at=Identity(G))
M = Sphere(2)
A = RotationAction(M, G)
# use a rigid motion for the sake of the example
φ = RigidMotion(A, ξ)
x = rand(M)
φ(x) # ξ
φ'(x) # linear operator Alg(G) → Alg(G)
integrate(φ, x) # solution of x'=φ(x)⋅x
0.1*φ # rescaled motion
```
"""

abstract type AbstractMotion{TA<:AbstractGroupAction{LeftAction}} end


"""
    get_action(m::AbstractMotion) :: AbstractGroupAction

The action for the given motion.
"""
get_action(m::AbstractMotion) = m.A

#--------------------------------
# AbstractMotion Interface
#--------------------------------
"""
    get_dynamics(::AbstractMotion, x) :: Alg(G)

The actual motion x -> φ(x) in Alg(G).

**Shortcut**: `m(x)` ≡ `get_dynamics(m, x)`
"""
function get_dynamics end

@doc raw"""
    get_lin_at(φ::AbstractMotion, ::Any) :: Function

The linear operator ``ξ ↦ ⟨dφ,ξ⟩_x+[φ(x),ξ]`` at the point ``x`` for motion ``φ``.
"""
function get_lin_at end
#--------------------------------


(m::AbstractMotion)(x) = get_dynamics(m, x)

Base.adjoint(m::AbstractMotion) = x -> get_lin_at(m, x)

abstract type AbstractAffineMotion{TA} <: AbstractMotion{TA} end

get_lin_at(m::AbstractAffineMotion, ::Any) = get_lin(m)

"""
Compute the tangent vector corresponding to the motion
at the point x on the manifold.
"""
Base.:*(m::AbstractMotion, x) = apply_diff_group(get_action(m), Identity(base_group(get_action(m))), m(x), x)

@doc raw"""
    struct AffineMotion <: AbstractAffineMotion
        A # action G -> Diff(M)
        f # motion: M -> Alg(G)
        lin # linear part: operator Alg(G) ⊗ Alg(G)*
    end

General affine motion, defined from an action of a group ``G``
on a manifold ``M``,
a dynamics ``f \colon G \to \mathfrak{g}``,
 and the corresponding constant linear part `lin`,
a linear endormorphism of ``\mathfrak{g}``.
"""
struct AffineMotion{TA<:AbstractGroupAction{LeftAction},TF,TL} <: AbstractAffineMotion{TA}
    A::TA # action G -> Diff(M)
    f::TF # motion: M -> Alg(G)
    lin::TL # linear part: operator Alg(G) ⊗ Alg(G)*
end

Base.show(io::IO, m::AffineMotion) = print(io, "AffineMotion($(m.A), [...])")

get_dynamics(m::AffineMotion, x) = m.f(x)

@doc raw"""
    get_lin(m::AffineMotion) :: Function

The linear part of the motion, a linear endomorphism of ``\mathfrak{G}``.
"""
get_lin(m::AffineMotion) = m.lin


Base.:+(m1::AbstractAffineMotion{TA}, m2::AbstractAffineMotion{TA})  where {TA} = _add_affine_motions(m1,m2)

function _add_affine_motions(m1::AbstractAffineMotion{TA}, m2::AbstractAffineMotion{TA})  where {TA}
    # assert_equal_actions(m1, m2, "Cannot add motion with different actions")
    a1 = get_action(m1)
    function f(u)
        return get_dynamics(m1, u) + get_dynamics(m2, u)
    end
    l1 = get_lin(m1)
    l2 = get_lin(m2)
    function lin(ξ)
        return l1(ξ) + l2(ξ)
    end
    return AffineMotion(a1, f, lin)
end


function Base.:*(s::Number, m::AbstractAffineMotion)
    A = get_action(m)
    lin = get_lin(m)
    return AffineMotion(A, x -> s*m(x), ξ -> s*lin(ξ))
end

Base.:-(m::AbstractAffineMotion) = -1*m


# TODO: is it ever useful? what about the dual version?
function lift_at(m::AbstractAffineMotion, x)
    A = get_action(m)
    return AffineMotion(
        GroupOperationAction(base_group(A)),
        χ -> m(apply(A, χ, x)),
        get_lin(m)
    )
end


# Integration

function _prepare_problem(
    G, # Group
    dyn,
    method,
    )
    u0 = identity_element(G)
    diffop = ManifoldDiffEq.LieManifoldDiffEqOperator{Float64}(dyn)
    prob = ManifoldDiffEq.ManifoldODEProblem(diffop, u0, (0, 1.0), G)
    # new:
    A = GroupOperationAction(G)
    algo = method(G, Manifolds.GroupExponentialRetraction(), A)
    return prob, algo
end

# function integrate_problem(prob, algo, dt, method)
#     # alg = ManifoldDiffEq.ManifoldLieEuler(M, ExponentialRetraction(), action)
#     sol = OrdinaryDiffEq.solve(prob, algo, dt=dt)
#     return sol
# end


"""
    integrate_lift(m::AbstractMotion, x0) :: AbstractODESolution

Integrate the lifted motion in the acting group.
"""
function integrate_lift(
    m::AbstractMotion,
    x0,
    dt,
    method=ManifoldDiffEq.RKMK4
    )
    action = get_action(m)
    dyn(χ,p,t) = m(apply(action, χ, x0))
    # prob, algo = _prepare_problem(action, dyn, method)
    prob, algo = _prepare_problem(base_group(action), dyn, method)
    # return integrate_problem(prob, algo, dt, method)
    return OrdinaryDiffEq.solve(prob, algo, dt=dt)
end

"""
    integrate(motion, x0::TM) :: TM

Integrate the motion `motion` from the point `x0` on the manifold.
"""
function integrate(
    motion::AbstractMotion,
    x0,
    dt=0.1
) 
    action = get_action(motion)
    # manifold = group_manifold(action)
    # @assert is_point(manifold, x0)

    sol = integrate_lift(motion, x0, dt)
    χ = sol(1.0)
    x = apply(action, χ, x0)
    return x
end


"""
 Compute matrix for Dφ in the given basis
"""
function get_lin_mat(m::AbstractMotion, x, B::AbstractBasis)
    G = base_group(get_action(m))
    op = get_lin_at(m, x)
    return matrix_from_lin_endomorphism(G, op, B)
end



"""
Compose the adjoint action with the matrix M, i.e.,
given M ∈ Alg(G) ⊗ Alg(G)*,
the new operator is defined by
ξ ↦ χ (Mξ) χ⁻¹
and the function returns the corresponding matrix.
"""
function compose_adjoint(
    G,
    elt,
    morph_mat,
    B::AbstractBasis
    ) 
    op(ξ) = adjoint_action(G, elt, ξ)
    return compose_lie_matrix_op(G, op, morph_mat, B)
end

@doc raw"""
     compute_morphism(φ::Motion, x, B::AbstractBasis)

Integrate the lift of the motion ``φ`` at the point ``x``,
this gives a group element ``χ``.
This allows to compute the associate morphism, i.e., the operator
```math
ξ ↦ χ^{-1} (\exp(Dφ)ξ) χ
```
"""
function compute_morphism(motion, x, B; dt=0.1)
    action = get_action(motion)

    sol = integrate_lift(motion, x, dt)
    χ = sol(1)


    G = base_group(action)
    mm = get_lin_mat(motion, x, B)
    morph = compose_adjoint(G, inv(G, χ), exp(mm), B)

    return χ, morph
end

include("Motion/Rigid.jl")
include("Motion/Translation.jl")
include("Motion/MultiAffine.jl")
include("Motion/FlatAffine.jl")
include("Motion/GroupMotion.jl")




