"""
Simple Bradley-Terry rating fitter (order-independent)

Features:
- Accepts a list of match results (winner, loser) with no draws.
- Aggregates pairwise counts so results are independent of input order or batching.
- Fits strengths p_i (positive) using Hunter's MM updates (guaranteed to increase likelihood).
- Returns log-strengths (can be scaled to Elo-like ratings if desired).

Usage example (run this file): demonstrates commutativity of input order.

References:
- Bradley, T. A. (1952). "Rank Analysis of Incomplete Block Designs" (model origin)
- Hunter, D. R. (2004). "MM algorithms for generalized Bradley-Terry models"
"""
from math import log
from collections import defaultdict
import math
import csv


def aggregate_matches(matches):
    """Aggregate a list of matches into pairwise counts.

    matches: iterable of (winner, loser) identifiers (hashable)

    Returns:
      players: set of all player ids
      w: dict of dict where w[i][j] = times i beat j
      n: dict of dict where n[i][j] = total matches between i and j
    """
    players = set()
    w = defaultdict(lambda: defaultdict(int))
    n = defaultdict(lambda: defaultdict(int))
    for winner, loser in matches:
        players.add(winner)
        players.add(loser)
        w[winner][loser] += 1
        n[winner][loser] += 1
        n[loser][winner] += 1
    return players, w, n


def fit_bradley_terry(matches, max_iter=10000, tol=1e-12, verbose=False):
    """Fit Bradley-Terry strengths from match list.

    The implementation uses Hunter's MM updates (multiplicative update
    for positive strengths `p`). The algorithm depends only on aggregated
    pairwise counts, therefore it's invariant to the order or batching of
    input matches.

    Args:
      matches: iterable of (winner, loser)
      max_iter: maximum iterations for MM updates
      tol: relative tolerance for convergence (max change / p)
      verbose: print progress

    Returns:
      ratings: dict mapping player -> log-strength (float). These are
               identifiable only up to additive constant. The function
               normalizes by setting mean(log p)=0.
      strengths: dict mapping player -> p (positive strengths)
    """
    players, w, n = aggregate_matches(matches)
    players = sorted(players)
    if not players:
        return {}, {}

    # total wins per player
    t = {i: sum(w[i].values()) for i in players}

    # initialize positive strengths
    p = {i: 1.0 for i in players}

    # detect isolated players (no matches at all)
    isolated = [i for i in players if t[i] == 0 and not any(n[i].values())]
    if isolated:
        # keep them at p=1.0 (they won't affect others). We still include them.
        if verbose:
            print(f"Warning: isolated players with no matches: {isolated}")

    for iteration in range(1, max_iter + 1):
        max_rel_change = 0.0
        p_new = {}
        # compute denominators and new p
        for i in players:
            denom = 0.0
            # sum over opponents j with any matches between i and j
            for j, nij in n[i].items():
                if nij <= 0:
                    continue
                denom += nij / (p[i] + p[j])
            # If denom == 0 it means player i had no paired matches (isolated)
            if denom > 0:
                p_i_new = t[i] / denom
            else:
                # leave as-is (no information)
                p_i_new = p[i]
            p_new[i] = p_i_new
        # normalize strengths to fix scale (geometric mean -> mean(log p)=0)
        # compute geometric mean
        # Add small epsilon to avoid log(0) or division issues
        epsilon = 1e-300
        for i in players:
            p_new[i] = max(p_new[i], epsilon)
        
        log_sum = sum(math.log(v) for v in p_new.values())
        mean_log = log_sum / len(p_new)
        factor = math.exp(-mean_log)
        for i in players:
            p_new[i] *= factor
            rel_change = abs(p_new[i] - p[i]) / max(p[i], epsilon)
            if rel_change > max_rel_change:
                max_rel_change = rel_change
        p = p_new
        if verbose and (iteration % 100 == 0 or iteration == 1):
            print(f"iter={iteration} max_rel_change={max_rel_change:.3e}")
        if max_rel_change < tol:
            if verbose:
                print(f"converged in {iteration} iterations")
            break

    # produce log-strength ratings normalized to zero mean
    logp = {i: math.log(v) for i, v in p.items()}
    mean_logp = sum(logp.values()) / len(logp)
    ratings = {i: logp[i] - mean_logp for i in players}

    return ratings, p


def ratings_to_elo(ratings, scale=400.0, base=10.0):
    """Convert log-strength ratings to an Elo-like scale.

    rating_elo = scale * log_base(rating_exp) where rating_exp = exp(logp)
    For consistency, ratings were produced as log p with zero mean. Convert back
    via Elo = scale * log_base(e^(rating)). Using base 10 by default yields
    Elo = scale * log10(e) * rating_log = 173.717 * rating_log when scale=400.
    """
    # ratings are log p with mean 0. Convert: Elo = scale * log_base(exp(1)) * rating
    factor = scale / math.log(base)
    return {i: factor * r for i, r in ratings.items()}


def log_base(x):
    return math.log(x)


def load_matches_from_csv(filepath, winner_col='winner', loser_col='loser', has_header=True):
    """Load match results from a CSV file.

    Args:
      filepath: path to CSV file
      winner_col: column name or index (int) for winner
      loser_col: column name or index (int) for loser
      has_header: whether the CSV has a header row

    Returns:
      list of (winner, loser) tuples
    """
    matches = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
        if not rows:
            return matches
        
        if has_header:
            header = rows[0]
            data_rows = rows[1:]
            # if column names provided, find indices
            if isinstance(winner_col, str):
                winner_idx = header.index(winner_col)
            else:
                winner_idx = winner_col
            if isinstance(loser_col, str):
                loser_idx = header.index(loser_col)
            else:
                loser_idx = loser_col
        else:
            data_rows = rows
            winner_idx = winner_col if isinstance(winner_col, int) else 0
            loser_idx = loser_col if isinstance(loser_col, int) else 1
        
        for row in data_rows:
            if len(row) > max(winner_idx, loser_idx):
                winner = row[winner_idx].strip()
                loser = row[loser_idx].strip()
                if winner and loser:
                    matches.append((winner, loser))
    
    return matches

def analyze_rating_convergence(matches, step_size=10, verbose=True):
    """Analyze how ratings converge as matches accumulate.

    Args:
      matches: list of (winner, loser) tuples
      step_size: number of matches to add at each step
      verbose: print progress

    Returns:
      list of dicts with keys: 'num_matches', 'max_rating_change', 'ratings'
    """
    if not matches:
        return []
    
    results = []
    prev_ratings = None
    
    for i in range(step_size, len(matches) + 1, step_size):
        subset = matches[:i]
        ratings, _ = fit_bradley_terry(subset, verbose=False)
        
        if prev_ratings is not None:
            # calculate max rating change from previous step
            common_players = set(ratings.keys()) & set(prev_ratings.keys())
            if common_players:
                changes = [abs(ratings[p] - prev_ratings[p]) for p in common_players]
                max_change = max(changes)
            else:
                max_change = float('inf')
        else:
            max_change = float('inf')
        
        results.append({
            'num_matches': i,
            'max_rating_change': max_change,
            'ratings': ratings.copy()
        })
        
        if verbose:
            print(f"Matches: {i:4d} | Max rating change: {max_change:10.6f} | Players: {len(ratings)}")
        
        prev_ratings = ratings
    
    # add final state if not already included
    if len(matches) % step_size != 0:
        ratings, _ = fit_bradley_terry(matches, verbose=False)
        if prev_ratings is not None:
            common_players = set(ratings.keys()) & set(prev_ratings.keys())
            if common_players:
                changes = [abs(ratings[p] - prev_ratings[p]) for p in common_players]
                max_change = max(changes)
            else:
                max_change = float('inf')
        else:
            max_change = float('inf')
        
        results.append({
            'num_matches': len(matches),
            'max_rating_change': max_change,
            'ratings': ratings.copy()
        })
        
        if verbose:
            print(f"Matches: {len(matches):4d} | Max rating change: {max_change:10.6f} | Players: {len(ratings)}")
    
    return results


if __name__ == '__main__':
    import sys
    import os
    
    # Demo 1: Order independence
    print("=" * 60)
    print("DEMO 1: Order Independence Test")
    print("=" * 60)
    demo_matches = [
        ('A', 'B'), ('A', 'B'), ('A', 'C'),
        ('B', 'C'), ('C', 'B'), ('D', 'A'), ('D', 'B')
    ]

    # shuffled order 1
    a_matches = demo_matches[:]
    # shuffled order 2 (reverse)
    b_matches = list(reversed(demo_matches))

    r1, p1 = fit_bradley_terry(a_matches, verbose=False)
    r2, p2 = fit_bradley_terry(b_matches, verbose=False)

    print("\nRatings (log-strength, mean=0) from original order:")
    for k, v in sorted(r1.items()):
        print(f"  {k}: {v:.6f}")

    print("\nRatings (log-strength) from reversed order:")
    for k, v in sorted(r2.items()):
        print(f"  {k}: {v:.6f}")

    # show max difference
    diffs = [abs(r1[k] - r2[k]) for k in r1]
    print(f"\nMax abs difference between orders: {max(diffs):.3e}")
    
    # Demo 2: CSV loading and convergence analysis
    print("\n" + "=" * 60)
    print("DEMO 2: CSV Loading and Convergence Analysis")
    print("=" * 60)
    
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_matches.csv')
    
    if os.path.exists(csv_path):
        print(f"\nLoading matches from: {csv_path}")
        matches = load_matches_from_csv(csv_path)
        print(f"Loaded {len(matches)} matches")
        
        print("\n" + "-" * 60)
        print("Convergence Analysis (step size=5):")
        print("-" * 60)
        convergence_data = analyze_rating_convergence(matches, step_size=5, verbose=True)
        
        # Find when rating changes become small (< 0.1)
        print("\n" + "-" * 60)
        print("Convergence Threshold Analysis:")
        print("-" * 60)
        thresholds = [0.5, 0.1, 0.05, 0.01]
        for threshold in thresholds:
            for data in convergence_data:
                if data['max_rating_change'] < threshold:
                    print(f"Rating change < {threshold}: achieved at {data['num_matches']} matches")
                    break
            else:
                print(f"Rating change < {threshold}: not achieved within {len(matches)} matches")
        
        # Show final ratings
        print("\n" + "-" * 60)
        print("Final Ratings (sorted by strength):")
        print("-" * 60)
        final_ratings = convergence_data[-1]['ratings']
        for player, rating in sorted(final_ratings.items(), key=lambda x: -x[1]):
            print(f"  {player:10s}: {rating:10.4f}")
    else:
        print(f"\nSample CSV not found at: {csv_path}")
        print("Skipping convergence analysis demo.")
