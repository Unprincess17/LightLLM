"""Path randomization for paired-block experiments.

Path order is randomized within each paired block (trial) to prevent
thermal drift, AVX downclock, GPU clock changes, or NIC state from
systematically favoring whichever path runs first.
"""
import random


def randomize_path_order(paths, seed=None):
    """Return a shuffled copy of paths.

    Args:
        paths: list of path names
        seed: random seed for reproducibility

    Returns: shuffled list (original list not modified)
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    return shuffled


def paired_block_order(cells, paths, seed=None):
    """Assign the same randomized path order to all cells in one paired block.

    Within one trial (block), all cells use the same path order. This ensures
    that thermal/clock drift affects all cells equally within the block.

    Args:
        cells: list of cell identifiers (e.g., (R, NM) tuples)
        paths: list of path names
        seed: random seed

    Returns: dict mapping cell -> ordered list of paths
    """
    order = randomize_path_order(paths, seed=seed)
    return {cell: list(order) for cell in cells}
