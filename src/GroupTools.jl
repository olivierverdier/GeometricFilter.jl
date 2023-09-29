
"""
The identity matrix on Lie algebra.
"""
function get_id_matrix_lie(G) 
    dim = manifold_dimension(G)
    T = ManifoldsBase.allocate_result_type(G, typeof(get_id_matrix_lie), ())
    return Matrix{T}(LinearAlgebra.I, dim, dim)
end


"""
Compose a matrix `mat` of a linear endomorphism of Alg(G)
in some basis `BG`` with an operator op : Alg(G) → T_p M
itself equipped with a basis `BM`.
"""
function compose_matrix_op(
    G, # group
    M,p, # Manifold, point
    op, # operator from Alg(G) to T_pM
    mat, # matrix in basis BG
    BG::AbstractBasis, # basis of Alg(G)
    BM::AbstractBasis, # basis of T_pM
    )
    idim = size(mat,2)
    odim = manifold_dimension(M)
    T = ManifoldsBase.allocate_result_type(G, typeof(compose_matrix_op), ())
    rmat = Array{T}(undef, odim, idim)
    for (b, rb) in zip(eachcol(mat), eachcol(rmat))
        vec = op(get_vector_lie(G, b, BG))
        rb[:] = get_coordinates(M, p, vec, BM)
    end
    return rmat
    # vecs = [op(get_vector_lie(G, b, BG)) for b in eachcol(mat)]
    # cols = [get_coordinates(M, p, vec, BM) for vec in vecs]
    # return hcat(cols...)
end

"""
Matrix of operator op : Alg(G) → T_p M
"""
function get_op_matrix(G, # group
                        M, p, # manifold + point
                        op, # operator from algebra to T_pM
                        BG::AbstractBasis, # Lie algebra basis
                        BM::AbstractBasis, # basis at T_pM
                        )
    return compose_matrix_op(G, M, p, op, get_id_matrix_lie(G), BG, BM)
    # basis = get_basis_lie(G, BG)
    # vecs = [op(b) for b in basis]
    # cols = [get_coordinates(M, p, vec, BM) for vec in vecs]
    # return hcat(cols...)
end


function compose_lie_matrix_op(
    G, # group
    op, # operator on Alg(G)
    mat, # matrix in the basis
    B::AbstractBasis # basis of Alg(G)
    )
    # basis = get_basis_lie(G, B)
    # vecs = [op(b) for b in basis]
    return compose_matrix_op(G, G, identity_element(G), op, mat, B, B)
    # vecs = [op(get_vector_lie(G, c, B)) for c in eachcol(mat)]
    # cols = [get_coordinates_lie(G, vec, B) for vec in vecs]
    # return hcat(cols...)
end



"""
Compute the matrix of an operator on a Lie algebra.
"""
function matrix_from_lin_endomorphism(
    G, # group
    op, # operator on Alg(G)
    B::AbstractBasis # basis of Alg(G)
    )
    return compose_lie_matrix_op(G, op, get_id_matrix_lie(G), B)
end

inverse_adjoint_action(G::AbstractDecoratorManifold, p, X) = adjoint_action(G, inv(G, p), X)



"""
Projection matrix from Alg(G) to T_xM
"""
function get_proj_matrix(A::AbstractGroupAction, x, BG, BM)
    G = base_group(A)
    M = group_manifold(A)
    idim = manifold_dimension(G)
    odim = manifold_dimension(M)
    T = ManifoldsBase.allocate_result_type(G, typeof(get_proj_matrix), ())
    rmat = Array{T}(undef, odim, idim)

    mat = get_id_matrix_lie(G)
    for (v, rv) in zip(eachcol(mat), eachcol(rmat))
        bvec = get_vector_lie(G, v, BG)
        mvec = apply_diff_group(A, Identity(G), bvec, x)
        rv[:] = get_coordinates(M, x, mvec, BM)
    end
    return rmat
end
