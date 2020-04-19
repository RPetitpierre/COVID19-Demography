from scipy.special import expit
import numpy as np

def compute_p_states(n_ages, mortality_multiplier=1):

    """2b. Construct transition probabilities between disease severities
    There are three disease states: mild, severe and critical.
    - Mild represents sub-hospitalization.
    - Severe is hospitalization.
    - Critical is ICU.

    The key results of this section are:
    - p_mild_severe: n_ages x 2 x 2 matrix. For each age and comorbidity state
        (length two bool vector indicating whether the individual has diabetes and/or
        hypertension), what is the probability of the individual transitioning from
        the mild to severe state.
    - p_severe_critical, p_critical_death are the same for the other state transitions.

    All of these probabilities are proportional to the base progression rate
    for an (age, diabetes, hypertension) state which is stored in p_death_target
    and estimated via logistic regression.
    """
    
    def age_to_interval(i):
        """Return the corresponding comorbidity age interval for a specific age.

        Args:
            i (int): age.

        Returns:
            int: index of interval containing i in intervals.
        """
        for idx, a in enumerate(intervals):
            if i >= a[0] and i < a[1]:
                return idx
        return idx

    p_mild_severe_cdc = np.zeros(n_ages)
    """n_ages vector: The probability of transitioning from the mild to
        severe state for a patient of age i is p_mild_severe_cdc[i]. We will match
        these overall probabilities.

    Source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
    Using the lower bounds for probability of hospitalization, since that's more
    consistent with frequency of severe infection reported in
    https://www.nejm.org/doi/full/10.1056/NEJMoa2002032 (at a lower level of age granularity).
    """
    p_mild_severe_cdc[0:20] = 0.016
    p_mild_severe_cdc[20:45] = 0.143
    p_mild_severe_cdc[45:55] = 0.212
    p_mild_severe_cdc[55:65] = 0.205
    p_mild_severe_cdc[65:75] = 0.286
    p_mild_severe_cdc[75:85] = 0.305
    p_mild_severe_cdc[85:] = 0.313
    
    overall_p_critical_death = 0.49
    """float: Probability that a critical individual dies. This does _not_ affect
    overall mortality, which is set separately, but rather how many individuals
    end up in critical state. 0.49 is from
    http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
    """

    #overall probability of progression from critical to severe
    #https://www.ecdc.europa.eu/sites/default/files/documents/RRA-sixth-update-Outbreak-of-novel-coronavirus-disease-2019-COVID-19.pdf
    #taking midpoint of the intervals
    overall_p_severe_critical = (0.15 + 0.2) / 2

    # go back to using CDC hospitalization rates as mild->severe
    severe_critical_multiplier = overall_p_severe_critical / p_mild_severe_cdc
    critical_death_multiplier = overall_p_critical_death / p_mild_severe_cdc

    # get the overall CFR for each age/comorbidity combination by running the logistic model
    """
    Mortality model. We fit a logistic regression to estimate p_mild_death from
    (age, diabetes, hypertension) to match the marginal mortality rates from TODO.
    The results of the logistic regression are used to set the disease severity
    transition probabilities.
    """
    c_age = np.loadtxt('c_age.txt', delimiter=',').mean(axis=0)
    """float vector: Logistic regression weights for each age bracket."""
    c_diabetes = np.loadtxt('c_diabetes.txt', delimiter=',').mean(axis=0)
    """float: Logistic regression weight for diabetes."""
    c_hyper = np.loadtxt('c_hypertension.txt', delimiter=',').mean(axis=0)
    """float: Logistic regression weight for hypertension."""
    intervals = np.loadtxt('comorbidity_age_intervals.txt', delimiter=',')

    p_death_target = np.zeros((n_ages, 2, 2))
    for i in range(n_ages):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                if i < intervals[0][0]:
                    p_death_target[i, diabetes_state, hyper_state] = 0
                else:
                    p_death_target[i, diabetes_state, hyper_state] = expit(
                        c_age[age_to_interval(i)] + diabetes_state * c_diabetes +
                        hyper_state * c_hyper)

    #p_death_target *= params['mortality_multiplier']
    #p_death_target[p_death_target > 1] = 1

    #calibrate the probability of the severe -> critical transition to match the
    #overall CFR for each age/comorbidity combination
    #age group, diabetes (0/1), hypertension (0/1)
    progression_rate = np.zeros((n_ages, 2, 2))
    p_mild_severe = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from mild to severe state."""
    p_severe_critical = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from severe to critical state."""
    p_critical_death = np.zeros((n_ages, 2, 2))
    """float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from critical to dead state."""

    for i in range(n_ages):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                progression_rate[i, diabetes_state, hyper_state] = (p_death_target[i, diabetes_state, hyper_state]
                                                                    / (severe_critical_multiplier[i]
                                                                       * critical_death_multiplier[i])) ** (1./3)
                p_mild_severe[i, diabetes_state, hyper_state] = progression_rate[i, diabetes_state, hyper_state]
                p_severe_critical[i, diabetes_state, hyper_state] = severe_critical_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
                p_critical_death[i, diabetes_state, hyper_state] = critical_death_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
    #no critical cases under 20 (CDC)
    p_critical_death[:20] = 0
    p_severe_critical[:20] = 0
    #for now, just cap 80+yos with diabetes and hypertension
    p_critical_death[p_critical_death > 1] = 1

    #scale up all transitions proportional to the mortality_multiplier parameter
    p_mild_severe *= mortality_multiplier**(1/3)
    p_severe_critical *= mortality_multiplier**(1/3)
    p_critical_death *= mortality_multiplier**(1/3)
    p_mild_severe[p_mild_severe > 1] = 1
    p_severe_critical[p_severe_critical > 1] = 1
    p_critical_death[p_critical_death > 1] = 1
    
    return p_mild_severe, p_severe_critical, p_critical_death