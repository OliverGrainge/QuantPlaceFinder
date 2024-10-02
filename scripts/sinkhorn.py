import torch 


def sinkhorn_knopp(A, num_iters=100, tol=1e-5):
    """
    Applies the Sinkhorn-Knopp algorithm to transform A into a doubly stochastic matrix.

    Args:
        A (torch.Tensor): Input non-negative matrix of shape (n, n).
        num_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        torch.Tensor: Doubly stochastic matrix.
        bool: Convergence flag.
    """
    # Ensure A is non-negative
    assert torch.all(A >= 0), "All elements of A must be non-negative."

    # Initialize
    D = A.clone()
    n, m = D.shape
    if n != m:
        raise ValueError("Input matrix must be square.")

    for i in range(num_iters):
        # Row normalization
        row_sum = D.sum(dim=1, keepdim=True)
        # Avoid division by zero
        row_sum = torch.where(row_sum == 0, torch.tensor(1.0, device=D.device), row_sum)
        D = D / row_sum

        # Column normalization
        col_sum = D.sum(dim=0, keepdim=True)
        # Avoid division by zero
        col_sum = torch.where(col_sum == 0, torch.tensor(1.0, device=D.device), col_sum)
        D = D / col_sum

        # Check for convergence
        row_diff = torch.abs(D.sum(dim=1) - 1.0).max()
        col_diff = torch.abs(D.sum(dim=0) - 1.0).max()
        if row_diff.item() < tol and col_diff.item() < tol:
            print(f"Converged in {i+1} iterations.")
            return D, True

    print(f"Did not converge within {num_iters} iterations.")
    return D, False


a = torch.randn(100, 100) + 1
a[a<0] = 0

a_normed, converged = sinkhorn_knopp(a)
print("=")
print(a_normed.sum(dim=1))
print("-")
print(a_normed.t().sum(dim=1))
