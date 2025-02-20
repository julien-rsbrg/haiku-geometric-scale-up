import jax
import jax.numpy as jnp


def random_walk(
        senders,
        receivers,
        walk_length,
        p: float = 1,
        q: float = 1,
        rng=jax.random.PRNGKey(42),
        num_nodes=None):
    """
    Random walk on the input graph
    """
    if num_nodes is None:
        num_nodes = jnp.max(jnp.maximum(senders, receivers)) + 1
    # for each node we generate a random walk
    walks = []
    for t in range(num_nodes):
        walks.append(_random_walk(senders, receivers, walk_length, p, q, t, rng))
    return jnp.array(walks)


def _random_neighbor(t, senders, receivers, rng=jax.random.PRNGKey(42)):
    """sample a random neighbor of t
    """
    # Get the neighbors of t
    neighbors = receivers[senders == t]
    #bool_index = jnp.where(senders == t, senders, t)
    #neighbors = jnp.where(bool_index, receivers, bool_index)

    # Sample a random neighbor
    return jax.random.choice(rng, neighbors, shape=(), replace=False)


def _is_neighbour(t, t_prime, senders, receivers):
    """check if t_prime is a neighbour of t
    """
    return jnp.any(jnp.logical_and(senders == t, receivers == t_prime))


# Adapted from: https://louisabraham.github.io/articles/node2vec-sampling.html
def _random_walk(senders, receivers, walk_length, p, q, t, rng):
    """sample a random walk starting from t
    """
    # Normalize the weights to compute rejection probabilities
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    # Initialize the walk
    walk = jnp.empty((walk_length + 1,), dtype=jnp.int32)
    walk = walk.at[0].set(t)
    walk = walk.at[1].set(_random_neighbor(t, senders, receivers, rng))
    for j in range(2, walk_length + 1): # we dont count the first node as part of the walk
        while True:
            new_node = _random_neighbor(walk[j - 1], senders, receivers, rng)
            r = jax.random.uniform(rng, shape=(), minval=0, maxval=1)
            if new_node == walk[j - 2]:
                if r < prob_0: # back to previous node
                    break
            elif _is_neighbour(walk[j - 2], new_node, senders, receivers):
                if r < prob_1: # distance 1
                    break
            elif r < prob_2: # distance 2
                break
        walk = walk.at[j].set(new_node)
    return walk
