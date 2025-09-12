from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

def compare_two_groups(group1, group2, group_names=('Group1','Group2'),
                               alpha=0.05, large_sample_threshold=30, alternative='two-sided'):
    
    print("=== Hypotheses ===")
    print(f"H0: Mean({group_names[0]}) = Mean({group_names[1]})")
    print(f"H1: Mean({group_names[0]}) != Mean({group_names[1]}) (alternative='{alternative}')\n")
    
    n1, n2 = len(group1), len(group2)
    
    print(f"Sample sizes: {group_names[0]}={n1}, {group_names[1]}={n2}\n")
    
    # --- Step 1: Check for large sample (CLT) ---
    if n1 >= large_sample_threshold and n2 >= large_sample_threshold:
        print("Large sample sizes → use t-test via CLT, skip Shapiro-Wilk\n")
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False, alternative=alternative)
        print("=== Welch's t-test Result ===")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < alpha:
            print("Decision: Reject H0")
        else:
            print("Decision: Do not reject H0")
        print("\n" + "="*50 + "\n")
        return
    
    # --- Step 2: Shapiro-Wilk test for normality ---
    normal_flags = {}
    print("=== Shapiro-Wilk Test for Normality ===")
    for name, data in zip(group_names, [group1, group2]):
        stat, p = shapiro(data)
        if p < alpha:
            print(f"{name}: W={stat:.4f}, p-value={p:.4f} → NOT normal (reject H0)")
            normal_flags[name] = False
        else:
            print(f"{name}: W={stat:.4f}, p-value={p:.4f} → looks normal (do not reject H0)")
            normal_flags[name] = True
    
    # --- Step 3: Decide which test to use ---
    if all(normal_flags.values()):
        print("\nBoth groups normal → check variances with Levene's test")
        stat, p_levene = levene(group1, group2)
        print(f"Levene W={stat:.4f}, p-value={p_levene:.4f}")
        
        print("\n=== Hypotheses ===")
        print(f"H0: Mean({group_names[0]}) = Mean({group_names[1]})")
        print(f"H1: Mean({group_names[0]}) != Mean({group_names[1]}) (alternative='{alternative}')\n")
        
        if p_levene < alpha:
            print("Variances unequal → use Welch's t-test")
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False, alternative=alternative)
        else:
            print("Variances equal → use standard independent t-test")
            t_stat, p_value = ttest_ind(group1, group2, equal_var=True, alternative=alternative)
        
        print("\n=== t-test Result ===")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < alpha:
            print("Decision: Reject H0")
        else:
            print("Decision: Do not reject H0")
    else:
        print("\nData not normal → use Mann–Whitney U test")
        
        print("\n=== Hypotheses ===")
        print(f"H0: The distributions of {group_names[0]} and {group_names[1]} are equal")
        print(f"H1: The distributions of {group_names[0]} and {group_names[1]} are not equal\n")
        
        u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        print("=== Mann–Whitney U Test Result ===")
        print(f"U-statistic: {u_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < alpha:
            print("Decision: Reject H0")
        else:
            print("Decision: Do not reject H0")
    
    print("\n" + "="*50 + "\n")
