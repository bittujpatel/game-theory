import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_game(payoff_matrix):
    # Convert to numpy array
    A = np.array(payoff_matrix)
    
    # Add a constant to make all entries strictly positive
    constant = abs(A.min()) + 1 
    A_pos = A + constant
    
    num_strategies = A.shape[0]
    
    # Objective: Minimize p1 + p2 + p3 (which corresponds to 1/V)
    c = np.ones(num_strategies)
    
    # Constraints: A_pos.T * p >= 1  ->  -A_pos.T * p <= -1
    A_ub = -A_pos.T
    b_ub = -np.ones(num_strategies)
    
    # Solve the linear program
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    
    if res.success:
        # The sum of p_i is 1/V
        v_transformed = 1.0 / np.sum(res.x)
        
        # Calculate original probabilities
        probabilities = res.x * v_transformed
        
        # Calculate original game value
        game_value = v_transformed - constant
        
        return np.round(probabilities, 4), round(game_value, 4)
    else:
        return None, None

# Original Payoff Matrix for Politician 1
# Rows: Politician 1 Strategies (S1, S2, S3)
# Cols: Politician 2 Strategies (S1, S2, S3)
payoff = [
    [0, -2,  2],
    [5,  4, -3],
    [2,  3, -4]
]

p1_probs, game_value = solve_zero_sum_game(payoff)

print("--- Game Theory Solution ---")
print(f"Politician 1 Optimal Strategy Probabilities (S1, S2, S3): {p1_probs}")
print(f"Expected Game Value (in thousands of votes): {game_value}")
