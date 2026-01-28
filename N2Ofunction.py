
"""
Created on Tue Feb 11 09:52:31 2025

@author: ahmed.attia
"""

import numpy as np

def N2OEmission(aboveN, belowN, minN, orgN, lechN, tier=1, EF_regional=None, co2e=True):
    """
    Calculate Direct and Indirect N2O emissions based on nitrogen inputs.
    
    Parameters:
        aboveN (array-like): Above-ground nitrogen content.
        belowN (array-like): Below-ground nitrogen content.
        minN (array-like): Mineral nitrogen input.
        orgN (array-like): Organic nitrogen input.
        lechN (array-like): Leached nitrogen.
        tier (int, optional): Tier for emission factor (default=1).
        EF_regional (float, optional): Regional emission factor for Tier 2.
        co2e (bool, optional): Whether to return CO2 equivalent emissions (default=True).
        
    Returns:
        np.ndarray: CO2 equivalent N2O emissions (if co2e=True) or N2O emissions (if co2e=False).
    """

    # Set emission factor based on tier
    if tier == 1:
        tier_ef = 0.01
    elif tier == 2 and EF_regional is not None:
        tier_ef = EF_regional
    else:
        tier_ef = 0.01  # Default emission factor

    # Convert inputs to NumPy arrays for vectorized operations
    aboveN = np.array(aboveN)
    belowN = np.array(belowN)
    minN = np.array(minN)
    orgN = np.array(orgN)
    lechN = np.array(lechN)

    # Ensure input lengths are consistent
    if not (len(aboveN) == len(belowN) == len(minN) == len(orgN) == len(lechN)):
        raise ValueError("All input arrays must be of the same length")

    # Residue nitrogen content
    NrootResidue = belowN
    Nstraw = aboveN
    FCR = NrootResidue + Nstraw
    FON = orgN
    FSN = minN  # Total N amount deployed in kg/ha

    # Emission factors
    EF5 = 0.0075  # Default IPCC indirect EF (kg N2O-N per kg N leached)
    EF4 = 0.001   # Default EF for volatilization
    EF1 = tier_ef # Tier-based direct EF

    # Additional parameters
    EMN2O = 44 / 28  # Conversion factor for N2O emissions
    FRAC_LEACH = 0.3
    FRAC_GASM = 0.2
    FRAC_GASF = 0.1

    # Indirect N2O emissions
    N2O_L = lechN * EMN2O * EF5
    N2O_ATD = (FSN * EF4 * FRAC_GASF + FON * EF4 * FRAC_GASM) * EMN2O
    N2O_indirect = N2O_L + N2O_ATD

    # Direct N2O emissions
    N2O_direct = (FSN + FON + FCR) * EF1 * EMN2O

    # Total N2O emissions
    N2O_total = N2O_indirect + N2O_direct

    # Convert to CO2 equivalent emissions
    co2_N2O = N2O_total * 273  # IPCC AR4 conversion factor

    return co2_N2O if co2e else N2O_total


aboveN = [5, 10, 15]
belowN = [3, 6, 9]
minN = [20, 30, 40]
orgN = [2, 4, 6]
lechN = [1, 2, 3]
tier = 2
EF_regional = 0.02

result = N2OEmission(aboveN, belowN, minN, orgN, lechN, tier, EF_regional, co2e=True)
print(result)  # Output: CO2 equivalent N2O emissions




