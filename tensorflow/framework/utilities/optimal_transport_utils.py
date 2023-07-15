import tensorflow as tf


@tf.function
def computeGrads(dy, X, mu, gamma):
    ''' X is the solution to the constrained optimal transport problem with
    X.shape = [batch_size, target_dim+1, source_dim] or [batch_size, target_dim+1, source_width, source_height]
    where X[:, 0] is the residual mass vector and X[:, 1:] is the transport plan to the target
    dy is the gradient wrt X
    retuns dC which is the gradient wrt cost matrix'''

    # jacobian computation

    # !TODO: explain computation logic (clearly write expressions and steps)
    # dx/dc = -gamma*I_bar@(D(x)-D(x)A^T@inv(H)@A@D(x)), I_bar = [0 I] for flattened x and c

    solution_shape = X.get_shape().as_list()
    if len(solution_shape) > 3:
        batch_size, target_dim = solution_shape[:2]
        source_dim = solution_shape[2] * solution_shape[3]
        source_axes = [-2, -1]
    else:
        batch_size, target_dim, source_dim = solution_shape
        source_axes = -1

    X_by_dLdX = X * dy  # D(x) @ dL/dx term

    r_by_dLdr = tf.expand_dims(X_by_dLdX[:, 0], axis=1)

    # precomputing some coeff.s and vectors
    r = tf.expand_dims(X[:, 0], axis=1)
    k_1 = 1.0 / (1.0 - mu - source_dim * tf.reduce_sum(tf.square(r), axis=source_axes, keepdims=True))

    row_sum = tf.reduce_sum(X_by_dLdX, axis=1, keepdims=True) # g_r + \sum_i g_pi ; where g = X_by_dLdX
    coeff = tf.subtract(
        tf.reduce_sum(r_by_dLdr, axis=source_axes, keepdims=True),
        source_dim * tf.reduce_sum(row_sum * r, axis=source_axes, keepdims=True))

    # gradient wrt beta
    dLdmu = - coeff * k_1

    common_vec = source_dim * (row_sum + dLdmu * r)

    # dLdC
    pre_dLdC = - gamma * (X_by_dLdX - common_vec * X)

    # gradient wrt binary cost
    dLdCb = pre_dLdC[:, 1:]

    # gradient wrt unary cost
    dLdCu = tf.expand_dims(pre_dLdC[:, 0], axis=1) - gamma * dLdmu * r

    return dLdCb, dLdCu, dLdmu

@tf.function
def computePartialOptimalTransportPlan(binary_costs, unary_costs, mu, gamma, max_it):
    # binary_costs: cost matrix tensor of shape = (batch_size, num_bins_in_target, num_bins_in_source)
    #               or (batch_size, num_bins_in_target, source_width, source_height)
    # unary_costs: cost vector tensor of shape = (batch_size, 1, num_bins_in_source) (or broadcastable to that shape)
    #               or (batch_size, 1, source_width, source_height)
    # mu: masses to be transported
    # 1 - mu amount of mass will be residual mass

    cost_shape = binary_costs.get_shape().as_list()
    if len(cost_shape) > 3:
        batch_size, target_dim = cost_shape[:2]
        source_dim = cost_shape[2] * cost_shape[3]
        source_axes = [-2, -1]
    else:
        batch_size, target_dim, source_dim = cost_shape
        source_axes = -1

    Kb = tf.exp(- gamma * binary_costs)
    Ku = tf.exp(- gamma * unary_costs)

    q = tf.constant(1. / source_dim, dtype=tf.float32)

    Kb_T_1 = tf.reduce_sum(Kb, axis=1, keepdims=True)

    #beta = tf.constant(beta, tf.float32)

    def fixedPointIteration(mu_):
        # takes mu_(n), returns mu_(n+1)

        p_ = q / (Ku + mu_ * Kb_T_1)
        return mu / tf.reduce_sum(p_ * Kb_T_1, axis=source_axes, keepdims=True)

    mu_0 = tf.ones_like(mu, dtype=tf.float32)
    k_0 = tf.constant(0, dtype=tf.int32)
    k, mu_ = (
        tf.while_loop(cond=lambda k, b: tf.less(k, max_it),
                      body=lambda k, b: (k + 1, fixedPointIteration(b)),
                      loop_vars=(k_0, mu_0),
                      shape_invariants=(k_0.get_shape(), mu_0.get_shape())
                      )
    )
    '''k, beta_ = tf.nest.map_structure(tf.stop_gradient,
        tf.while_loop(cond=lambda k, b: tf.less(k, max_it),
                      body=lambda k, b: (k + 1, fixedPointIteration(b)),
                      loop_vars=(k_0, beta_0),
                      shape_invariants=(k_0.get_shape(), tf.TensorShape([None, 1, 1]))))'''
    p = q / (Ku + mu_ * Kb_T_1)
    mu_ = mu / tf.reduce_sum(p * Kb_T_1, axis=source_axes, keepdims=True)
    X = tf.concat((Ku * p, mu_ * Kb * p), axis=1)

    return X

@tf.function
def partialOptimalTransportPlan(binary_costs, unary_costs, mu, gamma, max_it,
                                grad_method='inv'):

    # cost_matrix: cost matrix tensor of shape = (batch_size, num_bins_in_target, num_bins_in_source)
    if not grad_method in ['auto', 'inv']:
        raise ValueError('grad_method must be {} but got {}'.format(['auto', 'inv'], grad_method))

    if grad_method == 'auto':
        # normalize so that max cost is 1
        #nu = tf.stop_gradient(tf.reduce_max(cost_matrix, axis=[1, 2], keepdims=True))
        #X = computeConstrainedOptimalTransportPlan(cost_matrix / nu, source_masses, beta, gamma, max_it)
        X = computePartialOptimalTransportPlan(binary_costs, unary_costs, mu, gamma, max_it)
    else:
        def opt(c_b, c_u, mu_mass):
            # normalize so that max cost is 1
            nu = 1. #tf.maximum(tf.reduce_max(c, axis=[1, 2], keepdims=True), 5.)
            c_b = c_b / nu
            P = computePartialOptimalTransportPlan(c_b, c_u, mu_mass, gamma, max_it)
            grad_fn = lambda dy: (computeGrads(dy, P, mu, gamma/nu))
            return P, grad_fn

        X = tf.custom_gradient(opt)(binary_costs, unary_costs, mu)

    return X