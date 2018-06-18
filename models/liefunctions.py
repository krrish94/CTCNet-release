import numpy as np
import tensorflow as tf


# SE(3) exponential map
def SE3_expmap(vec):

    u = vec[:3]
    omega = vec[3:]

    theta = np.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

    omega_cross = np.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
    omega_cross = np.reshape(omega_cross, [3,3])

    A = np.sin(theta)/theta
    B = (1.0 - np.cos(theta))/(theta**2)
    C = (1.0 - A)/(theta**2)

    omega_cross_square = np.matmul(omega_cross, omega_cross)

    R = np.eye(3,3) + A*omega_cross + B*omega_cross_square

    V = np.eye(3,3) + B*omega_cross + C*omega_cross_square
    Vu = np.matmul(V,np.expand_dims(u,1))

    T = np.concatenate([R, Vu], 1)
    T  = np.concatenate((T, np.array([[0.0, 0.0, 0.0, 1.0]])), 0)

    return T


# SE(3) logarithm map
def SE3_logmap(T):

    R = T[:3,:3]
    t = T[:3,3]

    trace = R[0,0] + R[1,1] + R[2,2]
    trace = np.clip(trace, 0.0, 2.99999)
    theta = np.arccos((trace - 1.0)/2.0)
    omega_cross = (theta/(2*np.sin(theta)))*(R - np.transpose(R))

    xi = np.stack([t[0], t[1], t[2], omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]])

    return xi

def exponential_map_single(vec):

    "Decoupled for SO(3) and translation t"

    with tf.name_scope("Exponential_map"):
        # t = tf.expand_dims(vec[:3], 1)
        u = vec[:3]
        omega = vec[3:]

        theta = tf.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

        omega_cross = tf.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
        omega_cross = tf.reshape(omega_cross, [3,3])

        #Taylor's approximation for A,B and C not being used currently, approximations preferable for low values of theta
        # A = 1.0 - (tf.pow(theta,2)/factorial(3.0)) + (tf.pow(theta, 4)/factorial(5.0))
        # B = 1.0/factorial(2.0) - (tf.pow(theta,2)/factorial(4.0)) + (tf.pow(theta, 4)/factorial(6.0))
        # C = 1.0/factorial(3.0) - (tf.pow(theta,2)/factorial(5.0)) + (tf.pow(theta, 4)/factorial(7.0))

        A = tf.sin(theta)/theta

        B = (1.0 - tf.cos(theta))/(tf.pow(theta,2))

        C = (1.0 - A)/(tf.pow(theta,2))

        omega_cross_square = tf.matmul(omega_cross, omega_cross)

        R = tf.eye(3,3) + A*omega_cross + B*omega_cross_square

        V = tf.eye(3,3) + B*omega_cross + C*omega_cross_square
        Vu = tf.matmul(V,tf.expand_dims(u,1))

        T = tf.concat([R, Vu], 1)
        T  = tf.concat([T, tf.constant(np.array([[0.0, 0.0, 0.0, 1.0]]), dtype = tf.float32)], 0)

        return T


#6D pose vector
def tf_matrix_to_xi(T):

    R = T[:3,:3]
    t = T[:3,3]

    trace = R[0,0] + R[1,1] + R[2,2]
    trace = tf.clip_by_value(trace, 0.0, 2.99999)
    theta = tf.acos((trace - 1.0)/2.0)
    omega_cross = (theta/(2*tf.sin(theta)))*(R - tf.transpose(R))

    xi = tf.stack([t[0], t[1], t[2], omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]])

    return xi

#compose transformations
def omega_composition_multiple(transformation_vectors, seq_len):
    """
    todo: use skip indexing in while loop (will take some time), for combinations of products
    """

    Se3_matrices_batch = tf.map_fn(lambda x:exponential_map_single(transformation_vectors[x]), tf.range(0, seq_len, 1), tf.float32)

    idx = tf.constant(0)
    temp = Se3_matrices_batch[0]

    b = lambda i, m: [i+1, tf.matmul(m, Se3_matrices_batch[i+1])]
    c = lambda i, m: i<seq_len - 1

    matrix_product = tf.while_loop(c, b, loop_vars=[idx, temp], shape_invariants=[idx.get_shape(), temp.get_shape()])

    product = matrix_product[1]
    return product
