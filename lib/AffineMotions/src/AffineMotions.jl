module AffineMotions

using Manifolds

import ManifoldDiffEq
import OrdinaryDiffEq

import ManifoldGroupUtils as GU

export AbstractMotion, AbstractAffineMotion,
    RigidMotion, TranslationMotion,
    FlatAffineMotion,
    ZeroMotion,
    get_flat_action,
    integrate,
    # compose_adjoint,
    swap_group_motion

include("Motion.jl")

include("../ext/FilterMultiAffineExt/FilterMultiAffineExt.jl")

end
