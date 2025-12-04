"""
Flask web application for Bradley-Terry rating analysis
Provides CSV upload and rating visualization
"""
from flask import Flask, render_template, request, jsonify
import os
import csv
import json
import base64
import math
from io import BytesIO, StringIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from bt_rating import load_matches_from_csv, fit_bradley_terry, analyze_rating_convergence
from bt_rating_bayesian import mcmc_bradley_terry

app = Flask(__name__)


def calculate_win_probabilities(ratings):
    """Calculate pairwise win probabilities for all players.
    
    Args:
        ratings: dict mapping player -> log-strength rating
    
    Returns:
        dict mapping player1 -> dict mapping player2 -> win probability
    """
    players = list(ratings.keys())
    win_probs = {}
    
    for p1 in players:
        win_probs[p1] = {}
        for p2 in players:
            if p1 == p2:
                win_probs[p1][p2] = 0.5  # tie with self
            else:
                # P(p1 beats p2) = 1 / (1 + exp(-(r1 - r2)))
                r1 = ratings[p1]
                r2 = ratings[p2]
                prob = 1.0 / (1.0 + math.exp(-(r1 - r2)))
                win_probs[p1][p2] = prob
    
    return win_probs
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_csv():
    """Handle CSV upload and compute ratings"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    try:
        # Read CSV from uploaded file
        content = file.read().decode('utf-8')
        csv_file = StringIO(content)
        reader = csv.reader(csv_file)
        rows = list(reader)
        
        if len(rows) < 2:
            return jsonify({'error': 'CSV must have at least a header and one data row'}), 400
        
        # Parse matches
        header = rows[0]
        winner_col = request.form.get('winner_col', 'winner')
        loser_col = request.form.get('loser_col', 'loser')
        
        try:
            winner_idx = header.index(winner_col)
            loser_idx = header.index(loser_col)
        except ValueError:
            return jsonify({'error': f'Columns {winner_col} or {loser_col} not found in CSV'}), 400
        
        matches = []
        for row in rows[1:]:
            if len(row) > max(winner_idx, loser_idx):
                winner = row[winner_idx].strip()
                loser = row[loser_idx].strip()
                if winner and loser:
                    matches.append((winner, loser))
        
        if not matches:
            return jsonify({'error': 'No valid matches found in CSV'}), 400
        
        # Compute ratings (point estimates)
        ratings, strengths = fit_bradley_terry(matches, verbose=False)
        
        # Compute Bayesian estimates with confidence intervals
        bayesian_results = mcmc_bradley_terry(
            matches,
            n_samples=5000,
            burn_in=1000,
            thin=2,
            verbose=False
        )
        
        # Analyze convergence
        step_size = max(1, len(matches) // 20)  # ~20 data points
        convergence_data = analyze_rating_convergence(matches, step_size=step_size, verbose=False)
        
        # Generate charts
        charts = generate_charts(ratings, convergence_data, bayesian_results)
        
        # Calculate win probabilities
        win_probabilities = calculate_win_probabilities(bayesian_results['mean'])
        
        # Prepare response data with confidence intervals
        ratings_list = sorted(
            [{
                'player': k,
                'rating': round(v, 4),
                'mean': round(bayesian_results['mean'][k], 4),
                'ci_lower': round(bayesian_results['ci_lower'][k], 4),
                'ci_upper': round(bayesian_results['ci_upper'][k], 4),
                'std': round(bayesian_results['std'][k], 4)
            } for k, v in ratings.items()],
            key=lambda x: -x['mean']
        )
        
        convergence_list = [
            {
                'num_matches': d['num_matches'],
                'max_change': round(d['max_rating_change'], 4) if d['max_rating_change'] != float('inf') else None
            }
            for d in convergence_data
        ]
        
        # Round win probabilities for response
        win_probs_rounded = {
            p1: {p2: round(prob, 4) for p2, prob in probs.items()}
            for p1, probs in win_probabilities.items()
        }
        
        return jsonify({
            'success': True,
            'num_matches': len(matches),
            'num_players': len(ratings),
            'ratings': ratings_list,
            'convergence': convergence_list,
            'win_probabilities': win_probs_rounded,
            'charts': charts
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


def generate_charts(ratings, convergence_data, bayesian_results=None):
    """Generate base64-encoded chart images"""
    charts = {}
    
    # Chart 1: Ratings with Confidence Intervals
    if bayesian_results:
        fig, ax = plt.subplots(figsize=(12, max(8, len(ratings) * 0.5)))
        
        # Sort by mean rating
        players_sorted = sorted(bayesian_results['mean'].items(), key=lambda x: -x[1])
        players = [p[0] for p in players_sorted]
        means = [bayesian_results['mean'][p] for p in players]
        ci_lower = [bayesian_results['ci_lower'][p] for p in players]
        ci_upper = [bayesian_results['ci_upper'][p] for p in players]
        
        # Calculate error bars (distance from mean to CI bounds)
        errors_lower = [means[i] - ci_lower[i] for i in range(len(players))]
        errors_upper = [ci_upper[i] - means[i] for i in range(len(players))]
        
        y_pos = np.arange(len(players))
        
        # Plot horizontal bars with error bars
        colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in means]
        ax.barh(y_pos, means, color=colors, alpha=0.7, height=0.6)
        ax.errorbar(means, y_pos, xerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(players, fontsize=11)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
        ax.set_xlabel('Log-Strength Rating', fontsize=13, fontweight='bold')
        ax.set_ylabel('Player (Ranked)', fontsize=13, fontweight='bold')
        ax.set_title('Player Rankings with 95% Confidence Intervals', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add rank numbers
        for i, (player, mean) in enumerate(players_sorted):
            rank_x = min(means) - (max(means) - min(means)) * 0.15
            ax.text(rank_x, i, f'#{i+1}', 
                   fontsize=12, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        charts['ratings_confidence'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
    
    # Chart 2: Simple Bar Chart (original)
    fig, ax = plt.subplots(figsize=(10, 6))
    players = sorted(ratings.items(), key=lambda x: -x[1])
    player_names = [p[0] for p in players]
    rating_values = [p[1] for p in players]
    
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in rating_values]
    ax.barh(player_names, rating_values, color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Rating (Log-strength)', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    ax.set_title('Final Player Ratings', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['ratings_bar'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Chart 3: Convergence Line Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    num_matches = [d['num_matches'] for d in convergence_data]
    max_changes = [d['max_rating_change'] for d in convergence_data]
    
    # Filter out infinity values for plotting
    filtered_data = [(n, c) for n, c in zip(num_matches, max_changes) if c != float('inf')]
    if filtered_data:
        num_matches_filtered, max_changes_filtered = zip(*filtered_data)
        ax.plot(num_matches_filtered, max_changes_filtered, marker='o', linewidth=2, markersize=6, color='#3498db')
        ax.set_xlabel('Number of Matches', fontsize=12)
        ax.set_ylabel('Max Rating Change', fontsize=12)
        ax.set_title('Rating Convergence Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add threshold lines
        thresholds = [0.5, 0.1, 0.01]
        colors_thresh = ['orange', 'green', 'red']
        for thresh, color in zip(thresholds, colors_thresh):
            ax.axhline(y=thresh, color=color, linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold: {thresh}')
        ax.legend()
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['convergence_line'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Chart 4: Rating evolution over matches
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Track each player's rating over time
    all_players = set(ratings.keys())
    player_history = {player: [] for player in all_players}
    match_counts = []
    
    for data in convergence_data:
        match_counts.append(data['num_matches'])
        current_ratings = data['ratings']
        for player in all_players:
            player_history[player].append(current_ratings.get(player, 0))
    
    # Plot lines for each player
    for player in sorted(all_players):
        ax.plot(match_counts, player_history[player], marker='o', label=player, linewidth=2, markersize=4)
    
    ax.set_xlabel('Number of Matches', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Rating Evolution by Player', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['rating_evolution'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return charts


if __name__ == '__main__':
    import sys
    
    # Check if running in production mode
    if '--production' in sys.argv or os.environ.get('FLASK_ENV') == 'production':
        print("Error: For production, use gunicorn instead:")
        print("  gunicorn -w 4 -b 0.0.0.0:5000 app:app")
        sys.exit(1)
    
    print("Starting Bradley-Terry Rating Web App...")
    print("Open your browser to: http://localhost:5000")
    print("")
    print("Note: This is a development server.")
    print("For production deployment, use gunicorn.")
    app.run(debug=True, host='0.0.0.0', port=5000)
