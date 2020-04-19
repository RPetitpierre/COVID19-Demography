import numpy as np
import csv

def read_ages_contact_matrix(country, n_ages):
    """Create a country-specific contact matrix from stored data.

    Read a stored contact matrix based on age intervals. Return a matrix of
    expected number of contacts for each pair of raw ages. Extrapolate to age
    ranges that are not covered.

    Args:
        country (str): country name.

    Returns:
        float n_ages x n_ages matrix: expected number of contacts between of a person
            of age i and age j is Poisson(matrix[i][j]).
    """
    
    contact_matrix_age_groups_dict = {
    'infected_1': '0-4', 'contact_1': '0-4', 'infected_2': '5-9',
    'contact_2': '5-9', 'infected_3': '10-14', 'contact_3': '10-14',
    'infected_4': '15-19', 'contact_4': '15-19', 'infected_5': '20-24',
    'contact_5': '20-24', 'infected_6': '25-29', 'contact_6': '25-29',
    'infected_7': '30-34', 'contact_7': '30-34', 'infected_8': '35-39',
    'contact_8': '35-39', 'infected_9': '40-44', 'contact_9': '40-44',
    'infected_10': '45-49', 'contact_10': '45-49', 'infected_11': '50-54',
    'contact_11': '50-54', 'infected_12': '55-59', 'contact_12': '55-59',
    'infected_13': '60-64', 'contact_13': '60-64', 'infected_14': '65-69',
    'contact_14': '65-69', 'infected_15': '70-74', 'contact_15': '70-74',
    'infected_16': '75-79', 'contact_16': '75-79'}
    
    matrix = np.zeros((n_ages, n_ages))
    with open('Contact_Matrices/{}/All_{}.csv'.format(country, country), 'r') as f:
        csvraw = list(csv.reader(f))
    col_headers = csvraw[0][1:-1]
    row_headers = [row[0] for row in csvraw[1:]]
    data = np.array([row[1:-1] for row in csvraw[1:]])
    for i in range(len(row_headers)):
        for j in range(len(col_headers)):
            interval_infected = contact_matrix_age_groups_dict[row_headers[i]]
            interval_infected = [int(x) for x in interval_infected.split('-')]
            interval_contact = contact_matrix_age_groups_dict[col_headers[j]]
            interval_contact = [int(x) for x in interval_contact.split('-')]
            for age_infected in range(interval_infected[0], interval_infected[1]+1):
                for age_contact in range(interval_contact[0], interval_contact[1]+1):
                    matrix[age_infected, age_contact] = float(data[i][j])/(interval_contact[1] - interval_contact[0] + 1)

    # extrapolate from 79yo out to 100yo
    # start by fixing the age of the infected person and then assuming linear decrease
    # in their number of contacts of a given age, following the slope of the largest
    # pair of age brackets that doesn't contain a diagonal term (since those are anomalously high)
    for i in range(interval_infected[1]+1):
        if i < 65: # 0-65
            slope = (matrix[i, 70] - matrix[i, 75])/5
        elif i < 70: # 65-70
            slope = (matrix[i, 55] - matrix[i, 60])/5
        elif i < 75: # 70-75
            slope = (matrix[i, 60] - matrix[i, 65])/5
        else: # 75-80
            slope = (matrix[i, 65] - matrix[i, 70])/5

        start_age = 79
        if i >= 75:
            start_age = 70
        for j in range(interval_contact[1]+1, n_ages):
            matrix[i, j] = matrix[i, start_age] - slope*(j - start_age)
            if matrix[i, j] < 0:
                matrix[i, j] = 0

    # fix diagonal terms
    for i in range(interval_infected[1]+1, n_ages):
        matrix[i] = matrix[interval_infected[1]]
    for i in range(int((100-80)/5)):
        age = 80 + i*5
        matrix[age:age+5, age:age+5] = matrix[79, 79]
        matrix[age:age+5, 75:80] = matrix[75, 70]
    matrix[100, 95:] = matrix[79, 79]
    matrix[95:, 100] = matrix[79, 79]

    return matrix