import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
from tqdm import tqdm

def analyze_failure_galaxies():
    """Deep dive into why specific galaxies fail both models"""
    
    print("="*70)
    print("FAILURE GALAXY INVESTIGATION")
    print("="*70)
    
    # The 19 galaxies that fail both UC and Partanen models
    failure_list = ['DDO154', 'IC4202', 'NGC3741', 'NGC4217', 'NGC5005', 
                    'NGC7814', 'UGC00731', 'UGC02916', 'UGC02953', 'UGC03205', 
                    'UGC03580', 'UGC05253', 'UGC06786', 'UGC06787', 'UGC08699', 
                    'UGC09133', 'UGC11914', 'UGCA442', 'UGCA444']
    
    # Extreme outliers needing special attention
    extreme_outliers = {
        'UGC02953': 262.0,  # Catastrophic χ²
        'UGC05253': 707.0,  # Catastrophic χ²
        'UGC06786': 5620.0, # Extreme failure
        'UGC06787': 4067.0  # Extreme failure
    }
    
    # Load metadata
    metadata_df = pd.read_csv('SPARC_Lelli2016_masses_cleaned.csv')
    metadata_df['Name_std'] = metadata_df['Name'].str.strip().str.upper()
    
    print("\n1. CHECKING FOR COMMON PATTERNS IN FAILURES:")
    print("-"*50)
    
    failure_properties = []
    for galaxy_name in failure_list:
        # Find in metadata
        meta = metadata_df[metadata_df['Name_std'] == galaxy_name.upper()]
        if not meta.empty:
            meta = meta.iloc[0]
            failure_properties.append({
                'name': galaxy_name,
                'Type': meta.get('Type', np.nan),
                'i': meta.get('i', np.nan),  # Inclination
                'Qual': meta.get('Qual', np.nan),  # Quality flag
                'MHI': meta.get('MHI', np.nan),
                'L3.6': meta.get('L3.6', np.nan),
                'Rdisk': meta.get('Rdisk', np.nan),
                'Distance': meta.get('Dist', np.nan),
                'Vflat': meta.get('Vflat', np.nan)
            })
    
    failure_df = pd.DataFrame(failure_properties)
    
    # Analyze patterns
    print("\nINCLINATION ANALYSIS:")
    low_inc = failure_df[failure_df['i'] < 40]
    if len(low_inc) > 0:
        print(f"  {len(low_inc)} galaxies with i < 40°: {list(low_inc['name'])}")
        print("  → Possible projection effects!")
    
    print("\nQUALITY FLAG ANALYSIS:")
    poor_quality = failure_df[failure_df['Qual'] > 2]
    if len(poor_quality) > 0:
        print(f"  {len(poor_quality)} galaxies with Qual > 2: {list(poor_quality['name'])}")
        print("  → Data quality issues likely!")
    
    print("\nMORPHOLOGY DISTRIBUTION:")
    print(f"  Mean Hubble Type: {failure_df['Type'].mean():.1f}")
    print(f"  Failures by type:")
    for t_min, t_max, label in [(0, 3, 'Early'), (4, 6, 'Intermediate'), (7, 10, 'Late')]:
        count = len(failure_df[(failure_df['Type'] >= t_min) & (failure_df['Type'] <= t_max)])
        print(f"    {label} ({t_min}-{t_max}): {count} galaxies")
    
    print("\nGAS CONTENT:")
    gas_fraction = failure_df['MHI'] / (failure_df['MHI'] + failure_df['L3.6'] * 0.5)
    print(f"  Mean gas fraction: {gas_fraction.mean():.2f}")
    high_gas = failure_df[gas_fraction > 0.8]
    if len(high_gas) > 0:
        print(f"  {len(high_gas)} extremely gas-rich (>80%): {list(high_gas['name'])}")
    
    # Investigate extreme outliers
    print("\n2. EXTREME OUTLIER INVESTIGATION:")
    print("-"*50)
    
    for galaxy, chi2 in extreme_outliers.items():
        print(f"\n{galaxy} (χ² = {chi2}):")
        
        # Load rotation curve
        try:
            fname = f"sparc_data/{galaxy}_rotmod.dat"
            if os.path.exists(fname):
                df = pd.read_csv(fname, sep=r'\s+', comment='#',
                                names=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'])
                
                # Check for anomalies
                print(f"  Data points: {len(df)}")
                print(f"  R range: {df['R'].min():.1f} - {df['R'].max():.1f} kpc")
                print(f"  V_obs range: {df['V_obs'].min():.1f} - {df['V_obs'].max():.1f} km/s")
                
                # Check for non-monotonic behavior
                v_diff = np.diff(df['V_obs'])
                if np.any(v_diff < -20):  # Significant drops
                    print("  ⚠️ Non-monotonic rotation curve detected!")
                
                # Check for extreme velocities
                if df['V_obs'].max() > 400:
                    print("  ⚠️ Extremely high velocities (>400 km/s)")
                
                # Check disk dominance
                v_disk_frac = df['V_disk'].max() / df['V_obs'].max()
                if v_disk_frac > 2:
                    print(f"  ⚠️ V_disk >> V_obs (ratio = {v_disk_frac:.1f})")
                    
        except Exception as e:
            print(f"  Could not load data: {e}")
    
    # Statistical comparison with successes
    print("\n3. FAILURES VS SUCCESSES COMPARISON:")
    print("-"*50)
    
    # Load all galaxy properties
    all_galaxies = metadata_df[metadata_df['Name_std'].notna()]
    success_galaxies = all_galaxies[~all_galaxies['Name_std'].isin([f.upper() for f in failure_list])]
    
    # Compare distributions
    for prop in ['Type', 'i', 'Rdisk', 'MHI']:
        if prop in failure_df.columns and prop in success_galaxies.columns:
            fail_vals = failure_df[prop].dropna()
            success_vals = success_galaxies[prop].dropna()
            
            if len(fail_vals) > 0 and len(success_vals) > 0:
                # KS test for different distributions
                ks_stat, p_value = stats.ks_2samp(fail_vals, success_vals)
                print(f"\n{prop}:")
                print(f"  Failures: mean={fail_vals.mean():.2f}, std={fail_vals.std():.2f}")
                print(f"  Successes: mean={success_vals.mean():.2f}, std={success_vals.std():.2f}")
                print(f"  KS test: p={p_value:.3f} {'(significantly different)' if p_value < 0.05 else ''}")
    
    return failure_df

def test_specific_predictions():
    """Test specific theoretical predictions of the unified model"""
    
    print("\n" + "="*70)
    print("TESTING THEORETICAL PREDICTIONS")
    print("="*70)
    
    # 1. Test NGC 6503 prediction
    print("\n1. NGC 6503 PREDICTION TEST:")
    print("-"*50)
    print("Theory predicts: G_eff/G_N = 1.23")
    
    # Load NGC 6503 data
    try:
        df = pd.read_csv('sparc_data/NGC6503_rotmod.dat', sep=r'\s+', comment='#',
                        names=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'])
        
        # Calculate required enhancement
        v_bary = np.sqrt(df['V_gas']**2 + 0.5*df['V_disk']**2 + 0.7*df['V_bulge']**2)
        enhancement = (df['V_obs'] / v_bary)**2
        mean_enhancement = np.median(enhancement[df['R'] > 5])  # Outer regions
        
        print(f"  Observed enhancement: {mean_enhancement:.2f}")
        print(f"  Theory prediction: 1.23")
        print(f"  Agreement: {'✓' if abs(mean_enhancement - 1.23) < 0.1 else '✗'}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.plot(df['R'], df['V_obs'], 'ko-', label='Observed', markersize=4)
        plt.plot(df['R'], v_bary, 'b--', label='Baryonic', alpha=0.7)
        plt.plot(df['R'], v_bary * np.sqrt(1.23), 'r-', label='Theory (×1.23)', alpha=0.7)
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title('NGC 6503: Testing G_eff = 1.23 G_N')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(122)
        plt.plot(df['R'], enhancement, 'g-', linewidth=2)
        plt.axhline(1.23, color='r', linestyle='--', label='Theory')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('G_eff / G_N')
        plt.title('Gravity Enhancement Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ngc6503_prediction_test.png', dpi=150)
        plt.show()
        
    except Exception as e:
        print(f"  Could not test: {e}")
    
    # 2. Test morphology-energy correlation
    print("\n2. MORPHOLOGY-ENERGY CORRELATION TEST:")
    print("-"*50)
    
    metadata_df = pd.read_csv('SPARC_Lelli2016_masses_cleaned.csv')
    
    # Calculate binding energy for all galaxies
    valid = metadata_df[(metadata_df['Type'].notna()) & 
                        (metadata_df['MHI'].notna()) & 
                        (metadata_df['L3.6'].notna()) & 
                        (metadata_df['Rdisk'].notna())]
    
    M_bary = valid['MHI'] + 0.5 * valid['L3.6']
    log_E_g = np.log10(M_bary**2 / valid['Rdisk'])
    
    # Correlation test
    r, p = stats.pearsonr(valid['Type'], log_E_g)
    print(f"  Pearson r = {r:.3f}, p = {p:.2e}")
    print(f"  Theory expects: strong negative correlation")
    print(f"  Result: {'✓ CONFIRMED' if r < -0.7 and p < 0.001 else '✗ Not confirmed'}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(valid['Type'], log_E_g, alpha=0.6)
    z = np.polyfit(valid['Type'], log_E_g, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(0, 10, 100)
    plt.plot(x_fit, p_fit(x_fit), 'r--', label=f'r = {r:.3f}')
    plt.xlabel('Hubble Type')
    plt.ylabel('log(E_g) [arbitrary units]')
    plt.title('Morphology-Energy Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    # Binned averages
    for t in range(11):
        subset = valid[valid['Type'] == t]
        if len(subset) > 0:
            mean_E = np.mean(log_E_g[valid['Type'] == t])
            std_E = np.std(log_E_g[valid['Type'] == t])
            plt.errorbar(t, mean_E, yerr=std_E, fmt='o', capsize=5)
    plt.xlabel('Hubble Type')
    plt.ylabel('Mean log(E_g)')
    plt.title('Binned Morphology-Energy Relation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('morphology_energy_test.png', dpi=150)
    plt.show()
    
    # 3. Test scaling universality
    print("\n3. SCALING FACTOR UNIVERSALITY TEST:")
    print("-"*50)
    
    print("Testing if p₀ + p_color*T gives consistent scaling...")
    
    # For UC model: p = 1.12 - 0.08*T
    types = np.arange(0, 11)
    p_values = 1.12 - 0.08 * types
    
    print(f"  Early-type (T=0): p = {p_values[0]:.2f}")
    print(f"  Spiral (T=5): p = {p_values[5]:.2f}")
    print(f"  Late-type (T=10): p = {p_values[10]:.2f}")
    print(f"  Range: {p_values.min():.2f} - {p_values.max():.2f}")
    print(f"  Variation: {100*(p_values.max()/p_values.min() - 1):.1f}%")
    
    # 4. Cosmological predictions
    print("\n4. COSMOLOGICAL PREDICTIONS:")
    print("-"*50)
    
    predictions = {
        'H₀': (73.2, 0.8, 'km/s/Mpc'),
        'σ₈': (0.78, 0.02, ''),
        'S₈': (0.76, 0.02, ''),
        'GW speed': (1.0, 3e-16, 'c')
    }
    
    observations = {
        'H₀': (73.0, 1.0, 'SH0ES'),
        'σ₈': (0.776, 0.017, 'DES'),
        'S₈': (0.766, 0.020, 'KiDS'),
        'GW speed': (1.0, 7e-16, 'LIGO/Virgo')
    }
    
    print("\nParameter | Theory | Observation | Status")
    print("-"*50)
    for param in predictions:
        pred_val, pred_err, unit = predictions[param]
        obs_val, obs_err, source = observations[param]
        
        # Check consistency
        combined_err = np.sqrt(pred_err**2 + obs_err**2)
        sigma_diff = abs(pred_val - obs_val) / combined_err
        
        status = "✓" if sigma_diff < 2 else "✗"
        print(f"{param:10} | {pred_val:.3f}±{pred_err:.3f} | {obs_val:.3f}±{obs_err:.3f} ({source}) | {status}")
    
    return

# Run analyses
if __name__ == "__main__":
    failure_df = analyze_failure_galaxies()
    test_specific_predictions()