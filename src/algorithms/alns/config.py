from dataclasses import dataclass


@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm with tunable parameters."""

    # ===== STOPPING CRITERIA =====
    max_iterations: int = 10000
    max_time_seconds: float = 300.0  # 5 minutes
    max_iterations_without_improvement: int = 1000

    # ===== WEIGHT MANAGEMENT =====
    weight_update_period: int = 100  # Update weights every p iterations
    reaction_factor: float = 0.1  # γ ∈ [0, 1]; higher = faster adaptation towards successful destroy/repair operations

    # ===== DESTROY PARAMETERS =====
    min_removal_percentage: float = 0.10  # Remove at least 10% of served requests
    max_removal_percentage: float = 0.40  # Remove at most 40% of served requests

    # ===== SIMULATED ANNEALING PARAMETERS =====
    initial_temperature: float = 100.0
    cooling_rate: float = 0.99  # T = T * cooling_rate each iteration

    # ===== SUCCESS SCORING =====
    # Operators are rewarded based on solution quality to guide weight adaptation.
    # Higher scores incentivize operators that find better solutions.
    # The ratio (typically 5:1 to 13:1 in literature) balances quality vs. acceptance rate.
    #
    # Example: If operator used 10 times:
    #   - Finds 1 new best → score = 10, success_rate = 10/10 = 1.0
    #   - Gets 5 accepted  → score = 5,  success_rate = 5/10 = 0.5
    # This creates preference for quality improvements over mere acceptances.
    score_new_best: float = 10.0  # Reward for finding new global best
    score_accepted: float = 1.0   # Reward for accepted solution (not new best)
    # score_rejected is implicitly 0.0

    # ===== LOGGING =====
    log_interval: int = 100  # Print progress every N iterations
