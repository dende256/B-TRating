"""
Bayesian Bradley-Terry rating with confidence intervals
Uses Metropolis-Hastings MCMC to estimate posterior distributions
"""
import math
import numpy as np
from collections import defaultdict
from bt_rating import aggregate_matches


def log_posterior(log_ratings, wins, matches, prior_std=2.0):
    """
    Compute log posterior probability for Bradley-Terry model.
    
    Args:
      log_ratings: dict mapping player -> log(strength)
      wins: dict of dict where wins[i][j] = times i beat j
      matches: dict of dict where matches[i][j] = total matches between i and j
      prior_std: standard deviation of normal prior on log-ratings
    
    Returns:
      log probability (log likelihood + log prior)
    """
    log_prob = 0.0
    
    # Log likelihood
    for i in wins:
        for j in wins[i]:
            if matches[i][j] > 0:
                w_ij = wins[i][j]
                w_ji = matches[i][j] - w_ij
                
                # P(i beats j) = exp(r_i) / (exp(r_i) + exp(r_j))
                # Using log-sum-exp trick for numerical stability
                r_i = log_ratings[i]
                r_j = log_ratings[j]
                max_r = max(r_i, r_j)
                log_sum = max_r + math.log(math.exp(r_i - max_r) + math.exp(r_j - max_r))
                
                # Likelihood: w_ij * log(P(i beats j)) + w_ji * log(P(j beats i))
                log_prob += w_ij * (r_i - log_sum)
                log_prob += w_ji * (r_j - log_sum)
    
    # Log prior: N(0, prior_std^2) on each log-rating
    for player, rating in log_ratings.items():
        log_prob -= 0.5 * (rating / prior_std) ** 2
    
    return log_prob


def mcmc_bradley_terry(matches_list, n_samples=10000, burn_in=2000, thin=5, 
                       proposal_std=0.5, prior_std=2.0, verbose=False):
    """
    Estimate Bradley-Terry ratings with confidence intervals using MCMC.
    
    Args:
      matches_list: list of (winner, loser) tuples
      n_samples: number of MCMC samples to generate
      burn_in: number of initial samples to discard
      thin: keep every nth sample (reduces autocorrelation)
      proposal_std: standard deviation for Metropolis-Hastings proposals
      prior_std: standard deviation of prior on log-ratings
      verbose: print progress
    
    Returns:
      dict with keys:
        - 'mean': dict of player -> mean log-rating
        - 'median': dict of player -> median log-rating
        - 'std': dict of player -> standard deviation
        - 'ci_lower': dict of player -> 2.5th percentile (95% CI lower bound)
        - 'ci_upper': dict of player -> 97.5th percentile (95% CI upper bound)
        - 'samples': dict of player -> array of samples
    """
    players, wins, matches = aggregate_matches(matches_list)
    players = sorted(players)
    
    if not players:
        return {}
    
    # Initialize log-ratings at zero
    current_ratings = {p: 0.0 for p in players}
    current_log_prob = log_posterior(current_ratings, wins, matches, prior_std)
    
    # Storage for samples
    samples = {p: [] for p in players}
    
    accept_count = 0
    total_proposals = 0
    
    for iteration in range(n_samples + burn_in):
        # Propose new ratings by perturbing one player at a time
        # (Gibbs-like approach: update one component at a time)
        for player in players:
            # Propose new rating for this player
            proposed_ratings = current_ratings.copy()
            proposed_ratings[player] = current_ratings[player] + np.random.normal(0, proposal_std)
            
            # Compute log probability of proposal
            proposed_log_prob = log_posterior(proposed_ratings, wins, matches, prior_std)
            
            # Metropolis-Hastings acceptance
            log_accept_ratio = proposed_log_prob - current_log_prob
            
            if math.log(np.random.random()) < log_accept_ratio:
                current_ratings = proposed_ratings
                current_log_prob = proposed_log_prob
                accept_count += 1
            
            total_proposals += 1
        
        # Normalize ratings to have mean zero (identifiability constraint)
        mean_rating = sum(current_ratings.values()) / len(current_ratings)
        for p in players:
            current_ratings[p] -= mean_rating
        current_log_prob = log_posterior(current_ratings, wins, matches, prior_std)
        
        # Store samples after burn-in
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            for p in players:
                samples[p].append(current_ratings[p])
        
        if verbose and (iteration + 1) % 1000 == 0:
            accept_rate = accept_count / total_proposals if total_proposals > 0 else 0
            print(f"Iteration {iteration + 1}/{n_samples + burn_in}, "
                  f"Accept rate: {accept_rate:.2%}")
    
    # Compute statistics
    results = {
        'mean': {},
        'median': {},
        'std': {},
        'ci_lower': {},
        'ci_upper': {},
        'samples': {}
    }
    
    for p in players:
        arr = np.array(samples[p])
        results['mean'][p] = float(np.mean(arr))
        results['median'][p] = float(np.median(arr))
        results['std'][p] = float(np.std(arr))
        results['ci_lower'][p] = float(np.percentile(arr, 2.5))
        results['ci_upper'][p] = float(np.percentile(arr, 97.5))
        results['samples'][p] = arr
    
    if verbose:
        final_accept_rate = accept_count / total_proposals if total_proposals > 0 else 0
        print(f"\nFinal acceptance rate: {final_accept_rate:.2%}")
        print(f"Effective samples per player: {len(samples[players[0]])}")
    
    return results


if __name__ == '__main__':
    # Test with sample data
    from bt_rating import load_matches_from_csv
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_matches.csv')
    
    if os.path.exists(csv_path):
        print("Loading matches from sample_matches.csv...")
        matches = load_matches_from_csv(csv_path)
        print(f"Loaded {len(matches)} matches\n")
        
        print("Running Bayesian MCMC estimation...")
        print("(This may take a minute...)\n")
        
        results = mcmc_bradley_terry(
            matches, 
            n_samples=5000, 
            burn_in=1000, 
            thin=2,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("Bayesian Bradley-Terry Ratings with 95% Confidence Intervals")
        print("=" * 60)
        
        # Sort by mean rating
        sorted_players = sorted(results['mean'].items(), key=lambda x: -x[1])
        
        print(f"\n{'Rank':<6} {'Player':<12} {'Mean':<10} {'Median':<10} {'95% CI':<20} {'StdDev':<8}")
        print("-" * 76)
        
        for rank, (player, mean_rating) in enumerate(sorted_players, 1):
            median = results['median'][player]
            ci_lower = results['ci_lower'][player]
            ci_upper = results['ci_upper'][player]
            std = results['std'][player]
            ci_str = f"[{ci_lower:6.3f}, {ci_upper:6.3f}]"
            
            print(f"{rank:<6} {player:<12} {mean_rating:>9.4f}  {median:>9.4f}  {ci_str:<20} {std:>7.4f}")
        
        print("\n" + "=" * 60)
        print("Interpretation:")
        print("- Mean: Average rating across posterior samples")
        print("- 95% CI: We're 95% confident the true rating is in this range")
        print("- Wider CI = more uncertainty (fewer matches for that player)")
        print("=" * 60)
    else:
        print(f"Sample CSV not found at: {csv_path}")

