"""
Brandon Cunnane
Ride Hailing Case Study
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from math import e


class SimResult():
    """Collects the result of driver payment simulation"""
    def __init__(self):
        self.active_riders = []
        self.new_riders = []
        self.churn = []
        self.requests = []
        self.matches = []
        self.total_users = 0
        self.rides = 0
        self.profit = 0


def get_factorials(n):
    """"returns list of factorials from 0 to n"""
    factorials = [1]
    for i in range(1, n+1):
        factorials.append(factorials[i-1] * i)
    return factorials


def get_poisson_dist(n):
    """
    returns array of poisson distribution probabilities
    column index represents lambda
    row index represents number of occurances, x
    """
    poisson = np.zeros((n+1, n+1))
    for l in range(n+1):
        for x in range(n+1):
            poisson[x,l] = (l**x) * e**(-l) / facts[x]
    return poisson


def get_log_regr():
    """returns fitted logistic regression model"""
    # import data, split into training and test sets
    data = pd.read_table('driverAcceptanceData.csv',delimiter=',',index_col=0)
    x = data.PAY.to_numpy().reshape(-1,1)
    y = data.ACCEPTED.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    
    # fit model
    regr = LogisticRegression().fit(x_train, y_train)
    print(f'Logisitc Regression Score: {regr.score(x_test, y_test)}')
    
    # plot
    x_range = np.arange(0,50,5).reshape(-1,1)
    plt.plot(x_test, y_test, '.', label='Test Data')
    plt.plot(x_range, regr.predict_proba(x_range)[:,1], '--', label='LogRegr')
    plt.xlabel('Driver Pay Per Ride($)')
    plt.ylabel('Ride Acceptance')
    plt.yticks([0,1], ['NO','YES'])
    plt.legend(loc='center right')
    plt.savefig('Ride_hailing_logistic_regression_model.png', dpi=250)
    plt.show()
    return regr


def get_binomial_dist(p, n_max):
    """
    returns array of probabilities for binomial distribution
    row is number of successes, x
    col is number of trials, n
    
    p = probability of success on individual trial
    n_max = maximum number of trials to examine
    n = number of trials
    x = number of successes
    """
    binomial = np.zeros((n_max+1, n_max+1))
    for n in range(n_max+1):
        for x in range(n_max+1):
            C = facts[n]/(facts[n-x]*facts[x])
            binomial[x,n] = C * p**x * (1-p)**(n-x)
    return binomial


def apply_dist(num_users, dist):
    """returns array with occurances determined by the input distribution"""
    # initialize
    result = np.zeros(num_users, dtype=int)
    
    # calculate the number of occurances for specified number of users
    counts = dist * num_users
    counts = counts.round().astype(int)
    occurance_locs = np.where(counts>0)[0]
    if occurance_locs.size == 0:
        return result
    counts = counts[0:occurance_locs.max() + 1]
    
    # generate data that has specified number of users and occurances
    idx1,idx2 = (0,0)
    for n in range(len(counts)):
        idx2 = idx2 + counts[n]
        result[idx1:idx2] = n
        idx1 = idx2
    return result


def run_simulation():
    """determines the number of ride-hailing rides completed in 1 year"""
    # identify globals
    global poi_dist
    global bi_dist
    
    # constants
    MAX_RIDERS = 10000
    GROWTH_RATE = 0.2
    MAX_NEW = 1000
    
    # initiate simulation
    month = 1
    cols = ['prev_month','requested','this_month','total']
    rides = pd.DataFrame(columns=cols)
    
    sr = SimResult()
    
    # run simulation
    while sr.total_users < MAX_RIDERS and month <= 12:
        # add new riders
        sr.new_riders.append(int(rides.size * GROWTH_RATE) + 300)
        if sr.new_riders[-1] > MAX_NEW:
            sr.new_riders[-1] = MAX_NEW
        if sr.total_users + sr.new_riders[-1] > MAX_RIDERS:
            sr.new_riders[-1] = MAX_RIDERS - sr.total_users
        add_riders = pd.DataFrame(np.zeros((sr.new_riders[-1],4),dtype=int), columns=cols)
        add_riders.this_month = 1
        rides = pd.concat([rides, add_riders])
        
        # move this month to previous
        rides.prev_month = rides.this_month[:]
        rides.this_month = 0
        
        # get requests
        for i in range(1, rides.prev_month.max() + 1):
            prev = rides.loc[rides.prev_month == i, 'prev_month']
            rides.loc[rides.prev_month == i, 'requested'] = apply_dist(len(prev), poi_dist[:,i])
        sr.requests.append(rides.requested.sum())
        
        # get matches
        for i in range(1, rides.requested.max() + 1):
            reqs = rides.loc[rides.requested == i, 'requested']
            rides.loc[rides.requested == i, 'this_month'] = apply_dist(len(reqs), bi_dist[:,i])
        sr.matches.append(rides.this_month.sum())
        
        # drop riders with zero rides this month
        sr.churn.append(rides[rides.this_month == 0].size)
        rides = rides[rides.this_month != 0]
        sr.active_riders.append(rides.size)
        
        # sum total
        rides.total = rides.total + rides.this_month
        
        # finish month
        month +=1
        sr.total_users += sr.new_riders[-1]
    sr.rides = rides.total.sum()
    return sr


# def main():
#     # global variables to be used by multiple functions
#     global facts
#     global poi_dist
#     global bi_dist
    
# constants
RIDE_CHARGE = 30
CALC_LIMIT = 10

# setup
result = []
facts = get_factorials(CALC_LIMIT)
poi_dist = get_poisson_dist(CALC_LIMIT)
regr = get_log_regr()
pay = np.arange(25,29,.2).reshape(-1,1)
accept_probs = regr.predict_proba(pay)[:,1]

# run simulations
profit_per_ride = RIDE_CHARGE - pay[:,0]
for i in range(len(pay)):
    bi_dist = get_binomial_dist(accept_probs[i], CALC_LIMIT)
    result.append(run_simulation())
    result[-1].profit = int(result[-1].rides * profit_per_ride[i])
profits = [x.profit for x in result]

# plot results
plt.plot(pay, profits, label='profit')
plt.xlabel('Driver Pay Per Ride ($)')
plt.ylabel('12 Month Profit ($)')
plt.savefig('Driver_pay_vs_annual_profits.png', dpi=250)
plt.show()

# output ideal payment
max_idx = profits.index(max(profits))
print(f'Ideal payment: ${pay[max_idx,0]:.2f} per ride')
print(f'Annual profit: ${profits[max_idx]}')


# if __name__ == '__main__':
#     main()
