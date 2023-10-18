import jax
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as gaussian 
import einops
from jax.random import randint

def PF_objective_brute_force(flow, x, P, alpha=5.0): 
    """ Brute force implementation of the PF objective.
        Implemented for unbatched 1d inputs for simplicity 
    Inputs:
        flow  - Function that accepts an unbatched 1d input
                and returns a 1d output and the log determinant
        x     - Unbatched 1d input
        P     - List of numpy arrays that form a partition
                over range(x.size)
        alpha - Regularization hyperparameter
         
    Outputs:
        objective - PFs objective
    """
    # Evaluate log p(x) with a Gaussian prior 
    z, log_det = flow(x)
    log_pz = gaussian.logpdf(z, 0.0, 1.0).sum() 
    log_px = log_pz + log_det

    # Create the Jacobian matrix for every item in the batch
    G = jax.jacobian(lambda x: flow(x)[0])(x)

    # Compute Ihat_P
    Ihat_P = -log_det 
    for k in P:
        Gk = G[k,:]
        Ihat_P += 0.5*jnp.linalg.slogdet(Gk@Gk.T)[1]

    objective = -log_px + alpha*Ihat_P 
    return objective.mean()

def PF_objective_unbiased(flow, x, rng_key, alpha=5.0):
    """ Unbiased estimate of the PF objective when the partition size is 1

    Inputs: 
        flow    - Function that accepts an unbatched 1d input
                  and returns a 1d output and the log determinant
        x       - Unbatched 1d input
        rng_key - JAX random key
        alpha   -  Regularization hyperparameter
         
    Outputs:
        objective - PFs objective
    """
    # Evaluate log p(x) with a Gaussian prior and construct the vjp function 
    z, vjp, log_det = jax.vjp(flow, x, has_aux=True)
    log_pz = gaussian.logpdf(z, 0.0, 1.0).sum()
    log_px = log_pz + log_det

    # Sample an index in the partition
    z_dim = z.shape[-1]
    k = random.randint(rng_key , minval=0, maxval=z_dim , shape=(1,)) 
    k_onehot = (jnp.arange(z_dim) == k).astype(z.dtype)

    # Evaluate the k’th row of G and compute an unbiased estimate of Ihat_P
    Gk, = vjp(k_onehot)
    GkGkT = (Gk**2).sum()
    Ihat_P = -log_det + z_dim*0.5*jnp.log(GkGkT)

    objective = -log_px + alpha*Ihat_P 
    return objective.mean()



def iPF_objective_unbiased(flow, x, rng_key, gamma=10.0):
    """ Unbiased estimate of the iPF objective when the partition size is 1

    Inputs:
        flow    - Function that accepts an unbatched 1d input
                  and returns a 1d output and the log determinant x - Unbatched 1d input
        rng_key - JAX random key
        gamma   - Regularization hyperparameter
    Outputs:
        objective - iPFs objective
    """
    # Pass x through to the latent space and compute the prior 
    z, _ = flow(x)
    log_pz = gaussian.logpdf(z, 0.0, 1.0).sum()

    # Sample an index in the partition
    z_dim = z.shape[-1]
    k = random.randint(rng_key , minval=0, maxval=z_dim , shape=(1,)) 
    k_onehot = (jnp.arange(z_dim) == k).astype(z.dtype)

    # Compute the reconstruction and k’th row of J
    x_reconstr , Jk = jax.jvp(lambda x: flow(x, inverse=True)[0], (z,), (k_onehot ,))
    JkTJk = (Jk**2).sum()
    reconstruction_error = jnp.sum((x - x_reconstr)**2)

    # Compute the objective function
    objective = -log_pz + 0.5*jnp.log(JkTJk) + gamma*reconstruction_error 
    return objective.mean()


def construct_partition_mask(index, z_shape):
    """ In general, we can find the i'th row of a matrix A
        by computing A.T@mask where mask is zeros everywhere except
        at the i'th index where it is 1.
        
        This function finds all of the masks needed to find the rows
        in G that are in the index'th partition.
    
    Inputs:
        index   - Batched array of integers
        z_shape - Shape of the latent variable
    Outputs:
        masks - Array of masks to find the rows in G
    """
    batch_size , H, W, C = z_shape
    n_partitions = C   

    # The only non zero element of i'th row of 
    # partition_mask is at index[i]'th position
    # This is used to select a partition.
    # shape is (batch_size, C)
    partition_mask = jnp.arange(n_partitions) == index[:, None]

    # Create masks that will let us find the k’th rows of G using masked vjps.
    partition_size = H * W
    G_selection_mask = jnp.eye(partition_size)
    G_selection_mask = G_selection_mask.reshape((partition_size, H, W))

    # Put the masks together
    masks = jnp.einsum("bc,phw->pbhwc", partition_mask, G_selection_mask)
    return masks

def unbiased_objective_image(flow, x, rng_key, alpha=5.0, vectorized=True):
    """ PFs objective function for images. Number of partitions is given by number
        of channels of output.
    
    Inputs:
        flow    - Function that accepts a batched 3d input and returns a batched 3d output
                  and the log determinant
        x       - Batched 3d input with channel on the last axis
        rng_key - JAX random key
        alpha   - Regularization hyperparameter
        vectorized - Should all of the vjps be evaluated in parallel?
    Outputs:
        objective - PFs objective for images
    """
    # Assume that we partition over the last axis of z
    # and that x is a batched image with channel on the last axis
    batch_size, H, W, C = x.shape

    # Evaluate log p(x) and retrieve the function that 
    # lets us evaluate vector-Jacobian products
    z, _vjp, log_det = jax.vjp(flow, x, has_aux=True)
    vjp = lambda v: _vjp(v)[0]  # JAX convention to return a tuple
    log_pz = gaussian.logpdf(z, 0.0, 1.0).sum(axis=range(1, z.ndim))
    log_px = log_pz + log_det

    # Randomly sample the index of the partition we will evaluate
    n_partitions = z.shape[-1]
    index = randint(rng_key, minval=0, maxval=n_partitions, shape=(batch_size,))

    # Construct the masks that we’ll use to find the index’th partition of G.
    # masks.shape == (partition_size , batch_size , H, W, C)
    masks = construct_partition_mask(index, z.shape)

    # Evaluate the vjp each of the n_partition masks
    if vectorized:
        # This is memory intensive but fast
        Gk = jax.vmap(vjp)(masks)
    else:
        # This is slow but memory efficient
        Gk = jax.lax.map(vjp, masks)

    # Each element of GG^T is the dot product between rows of G
    # Construct GkGk^T and then take its log determinant
    Gk = einops.rearrange(Gk, "p b H W C -> b p (H W C)")
    GkGkT = jnp.einsum("bij,bkj->bik", Gk, Gk)
    Ihat_P = 0.5 * jnp.linalg.slogdet(Gk)[1] * n_partitions - log_det
    
    objective = -log_px + alpha * Ihat_P
    return objective.mean()
