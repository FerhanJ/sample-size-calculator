import streamlit as st
from scipy.stats import bernoulli
from statsmodels.stats import power as pwr 
import numpy as np

# write null and alternate hypothesis
# center page elements
st.title('Sample Size Calculator (Binary Vars)')
p0 = st.number_input('Baseline Rate (p0)', min_value=0.0, max_value=1.0, format='%.3f')
st.markdown(f'The rate in the control group is **{p0:.2%}**')
mde = st.number_input('Minimum Detectable Effect', min_value=0.001, max_value=1.0, step=0.001, format='%.3f', value=0.01)
st.markdown(f'The minimum detectable effect is **{mde:.2%}**')
control_to_test_ratio = st.number_input('Sample Size Ratio', min_value=0.0, value=1.0, step=0.1)
st.markdown(f'The control group is **{control_to_test_ratio:.2f}** times larger than the test group')

# sidebar elements 
with st.sidebar:
    st.title('Advanced Inputs')
    diff_under_null =  st.number_input(
        'Difference Under Null (H_0)', 
        format='%.3f', 
        max_value=1-p0, 
        help="Sample sizes are inaccurate when difference under null isn't 0 and sample size ratio is more than 1")
    st.markdown(f'{diff_under_null:.2%}')
    alternative = st.radio('Alternative', ('two-sided', 'smaller', 'larger'))
    alpha = st.slider('Type 1 Error', min_value=0.0, max_value=1.0,  value=0.05)
    beta = st.slider('Type 2 Error', min_value=0.0, max_value=1.0, value=0.2)

# calculate extra vars from inputs
p1 = p0 + diff_under_null
power = 1 - beta
pooled_var = (bernoulli.var(p0) + bernoulli.var(p1)) / 2

# sample size calculation 
test_sample_size = pwr.zt_ind_solve_power(
    effect_size = mde / np.sqrt(pooled_var),
    nobs1=None,
    alpha=alpha,
    power=power,
    ratio=control_to_test_ratio,
    alternative=alternative
)

# sample size section
control_sample_size = control_to_test_ratio * test_sample_size
control_sample_size = int(control_sample_size)
test_sample_size = int(test_sample_size)
sample_size_md = f'''
#### Sample Size
<font size="5"> Control Group: **{control_sample_size:,}** </font><br>
<font size="5"> Test Group: **{test_sample_size:,}** </font>
'''
st.markdown(sample_size_md, unsafe_allow_html=True)

st.markdown('### Statistical Formulation')
if alternative == 'two-sided':
    st.latex(f"H_{0}: p_{1} - p_{0} = {diff_under_null:.2f}")
    st.latex(f"H_{1}: p_{1} - p_{0} ≠ {diff_under_null:.2f}")
    beta_explanation_ineq = fr'${diff_under_null - mde:.3f} < p_{1} - p_{0} < {diff_under_null + mde:.3f}$'

elif alternative == 'larger':
    st.latex(f"H_{0}: p_{1} - p_{0} ≤ {diff_under_null:.2f}")
    st.latex(f"H_{1}: p_{1} - p_{0} > {diff_under_null:.2f}")
    beta_explanation_ineq = fr'$p_{1} - p_{0} < {diff_under_null + mde:.3f}$'

elif alternative == 'smaller':
    st.latex(f"H_{0}: p_{1} - p_{0} ≥ {diff_under_null:.2f}")
    st.latex(f"H_{1}: p_{1} - p_{0} < {diff_under_null:.2f}")
    beta_explanation_ineq = fr'$p_{1} - p_{0} > {diff_under_null - mde:.3f}$'

st.latex(f"p_{0} = {p0:.3f}")
st.markdown(fr'Where $p_{0}$ and $p_{1}$ are the control and test group probabilities respectively')
st.markdown(fr'If $H_{0}$ is true, it will be rejected (at most) **{alpha:.0%}** of the time')
st.markdown(fr'$H_{0}$ will not be rejected more than **{beta:.0%}** of the time if' + ' ' + beta_explanation_ineq)