import pandas as pd
from scipy import stats

def check_normality(series: pd.Series, max_sample_size: int = 5000) -> tuple:
    sample = series.dropna()
    if len(sample) > max_sample_size:
        sample = sample.sample(n=max_sample_size, random_state=42)
    stat, p_value = stats.shapiro(sample)
    return p_value, p_value > 0.05


def compare_two_groups(group1: pd.Series, group2: pd.Series, group1_name: str, group2_name: str) -> dict:
    p1, normal1 = check_normality(group1)
    p2, normal2 = check_normality(group2)

    result = {
        'normality': {
            group1_name: {'p_value': p1, 'is_normal': normal1},
            group2_name: {'p_value': p2, 'is_normal': normal2}
        }
    }

    if normal1 and normal2:
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        result['test'] = 't-тест Стьюдента (Welch)'
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        result['test'] = 'Mann-Whitney U'

    result.update({
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_group1': group1.mean(),
        'mean_group2': group2.mean()
    })
    return result


def test_correlation(series1: pd.Series, series2: pd.Series) -> dict:
    p1, normal1 = check_normality(series1)
    p2, normal2 = check_normality(series2)

    if normal1 and normal2:
        corr, p_value = stats.pearsonr(series1, series2)
        method = 'Пирсона'
        corr_type = 'линейная'
    else:
        corr, p_value = stats.spearmanr(series1, series2)
        method = 'Спирмена'
        corr_type = 'монотонная'

    abs_corr = abs(corr)
    if abs_corr > 0.7:
        strength = 'сильная'
    elif abs_corr > 0.3:
        strength = 'умеренная'
    else:
        strength = 'слабая'

    direction = 'положительная' if corr > 0 else 'отрицательная'

    return {
        'method': method,
        'corr_type': corr_type,
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'strength': strength,
        'direction': direction
    }


def chi_square_test(contingency_table: pd.DataFrame) -> dict:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected_min': expected.min(),
        'significant': p_value < 0.05,
        'contingency_table': contingency_table
    }