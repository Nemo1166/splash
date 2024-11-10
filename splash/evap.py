import math

import numpy as np
from solar import calc_daily_solar
from utils import dsin

def density_h2o(tc, pa):
    """
    Calculate density of water at 1 atm, g/cm^3
    
    This function calculates the temperature and pressure dependent density of
    pure water.
    
    Args:
        tc (ArrayLike): Air temperature, degrees C.
        pa (ArrayLike): Atmospheric pressure, Pa.
    
    Returns:
        float: Density of water, kg/m^3.
    """
    po = (0.99983952 +
          6.788260e-5  * tc +
          -9.08659e-6  * tc * tc +
          1.022130e-7  * tc * tc * tc +
          -1.35439e-9  * tc * tc * tc * tc +
          1.471150e-11 * tc * tc * tc * tc * tc +
          -1.11663e-13 * tc * tc * tc * tc * tc * tc +
          5.044070e-16 * tc * tc * tc * tc * tc * tc * tc +
          -1.00659e-18 * tc * tc * tc * tc * tc * tc * tc * tc)

    ko = (19652.17 +
          148.1830    * tc +
          -2.29995    * tc * tc +
          0.01281     * tc * tc * tc +
          -4.91564e-5 * tc * tc * tc * tc +
          1.035530e-7 * tc * tc * tc * tc * tc)

    ca = (3.26138 +
          (5.223e-4)  * tc +
          (1.324e-4)  * tc * tc +
          -(7.655e-7) * tc * tc * tc +
          (8.584e-10) * tc * tc * tc * tc)

    cb = ((7.2061e-5) +
          -(5.8948e-6)    * tc +
          (8.69900e-8)    * tc * tc +
          -(1.0100e-9)    * tc * tc * tc +
          (4.3220e-12)    * tc * tc * tc * tc)

    pbar = (1e-5) * pa

    pw = 1e3 * po * (ko + ca * pbar + cb * (pbar ** 2)) / (ko + ca * pbar + cb * (pbar ** 2) - pbar)
    return pw

def elv2pres(z, kG=9.80665, kL=0.0065, kMa=0.028963, kPo=101325, kR=8.31447, kTo=288.15):
    """
    Elevation to pressure

    Calculates atmospheric pressure for a given elevation.

    Args:
        z (float): Elevation, m.
        kG (float): Gravitational acceleration, m/s^2.
        kL (float): Adiabatic lapse rate, K/m.
        kMa (float): Molecular weight of dry air, kg/mol.
        kPo (float): Standard atmosphere, Pa.
        kR (float): Universal gas constant, J/mol/K.
        kTo (float): Base temperature, K.

    Returns:
        float: Atmospheric pressure for the given elevation, Pa.
    """
    return kPo * (1 - kL * z / kTo) ** (kG * kMa / (kR * kL))

def enthalpy_vap(tc):
    """
    Calculate enthalpy of vaporization

    This function calculates the temperature-dependent enthalpy of vaporization
    (latent heat of vaporization).

    Args:
        tc (float): Air temperature, degrees C.

    Returns:
        float: Enthalpy of vaporization, J/kg.
    """
    return 1.91846e6 * ((tc + 273.15) / (tc + 273.15 - 33.91)) ** 2

def psychro(tc, pa, kMa=0.028963, kMv=0.01802):
    """
    Calculate psychrometric constant

    This function calculates the temperature and pressure dependent
    psychrometric constant.

    Args:
        tc (float): Air temperature, degrees C.
        pa (float): Atmospheric pressure, Pa.
        kMa (float): Molecular weight of dry air, kg/mol.
        kMv (float): Molecular weight of water vapor, kg/mol.

    Returns:
        float: Psychrometric constant, Pa/K.
    """
    cp = specific_heat(tc)
    lv = enthalpy_vap(tc)
    return cp * kMa * pa / (kMv * lv)

def specific_heat(tc):
    """
    Calculate specific heat

    This function calculates the specific heat of moist air.

    Args:
        tc (float): Air temperature, degrees C.

    Returns:
        float: Specific heat of moist air, J/kg/K.
    """
    tc = np.clip(tc, 0, 100)
    cp = (1.0045714270 +
          (2.050632750e-3)  * tc -
          (1.631537093e-4)  * tc * tc +
          (6.212300300e-6)  * tc * tc * tc -
          (8.830478888e-8)  * tc * tc * tc * tc +
          (5.071307038e-10) * tc * tc * tc * tc * tc)
    return (1e3) * cp

def sat_slope(tc):
    """
    Calculate the temperature-dependent slope

    This function calculates the temperature-dependent slope of the saturation
    pressure temperature curve using the methodology presented in the eMast
    energy.cpp script.

    Args:
        tc (float): Air temperature, degrees C.

    Returns:
        float: Slope of saturation pressure temperature curve, Pa/K.
    """
    return (17.269) * (237.3) * (610.78) * np.exp(17.269 * tc / (237.3 + tc)) / (237.3 + tc) ** 2

def _validate_lat(lat):
    if isinstance(lat, float|int):
        if lat > 90 or lat < -90:
            raise ValueError(f"Warning: Latitude outside range of validity (should in [-90, 90])!")
    if isinstance(lat, np.ndarray):
        if lat.any() > 90 or lat.any() < -90:
            raise ValueError(f"Warning: Latitude outside range of validity (should in [-90, 90], got {lat})!")

def calc_daily_evap(lat, n, elv=0, y=0, sf=1, tc=23.0, sw=1.0, ke=0.01670, keps=23.44, komega=283, kw=0.26):
    """
    Calculate daily evaporation fluxes

    This function calculates daily radiation, condensation, and evaporation fluxes.

    Args:
        lat (float): Latitude, decimal degrees.
        n (float): Day of year.
        elv (float): Elevation, m A.S.L.
        y (float): Year.
        sf (float): Fraction of sunshine hours.
        tc (float): Mean daily air temperature, degrees C.
        sw (float): Evaporative supply rate, mm/hr.
        ke (float): Eccentricity of earth's orbit.
        keps (float): Obliquity of earth's elliptic.
        komega (float): Longitude of perihelion, degrees.
        kw (float): PET entrainment.

    Returns:
        dict: A dictionary containing calculated evaporation values.
    """
    pir = np.pi / 180

    _validate_lat(lat)
    if n < 1 or n > 366:
        raise ValueError(f"Warning: Day outside range of validity (should be [1, 366], got {n})!")

    evap = {}

    # Calculate radiation fluxes
    solar = calc_daily_solar(lat=lat, n=n, elv=elv, y=y, sf=sf, tc=tc, ke=ke, keps=keps, komega=komega)
    ru = solar['ru']
    rv = solar['rv']
    rw = solar['rw']
    rnl = solar['rnl_w.m2']
    hn = solar['hn_deg']
    rn_d = solar['rn_j.m2']
    rnn_d = solar['rnn_j.m2']
    evap['ra_j.m2'] = solar['ra_j.m2']
    evap['rn_j.m2'] = solar['rn_j.m2']
    evap['ppfd_mol.m2'] = solar['ppfd_mol.m2']

    # Calculate water-to-energy conversion (econ), m^3/J
    patm = elv2pres(elv)
    evap['patm_pa'] = patm

    s = sat_slope(tc)
    evap['s_pa.k'] = s

    lv = enthalpy_vap(tc)
    evap['lv_j.kg'] = lv

    pw = density_h2o(tc, patm)
    evap['pw_kg.m3'] = pw

    gam = psychro(tc, patm)
    evap['gam_pa.k'] = gam

    econ = s / (lv * pw * (s + gam))
    evap['econ_m3.j'] = econ

    # Calculate daily condensation (cn), mm
    cn = (1e3) * econ * np.abs(rnn_d)
    evap['cond_mm'] = cn

    # Estimate daily equilibrium evapotranspiration (eet_d), mm
    eet_d = 1e3 * econ * rn_d
    evap['eet_mm'] = eet_d

    # Estimate daily potential evapotranspiration (pet_d), mm
    pet_d = (1 + kw) * eet_d
    evap['pet_mm'] = pet_d

    # Calculate variable substitute (rx), (mm/hr)/(W/m^2)
    rx = (3.6e6) * (1 + kw) * econ
    evap['rx'] = rx

    # Calculate the intersection hour angle (hi), degrees
    cos_hi = sw / (rw * rv * rx) + rnl / (rw * rv) - ru / rv
    hi = np.where(cos_hi>=1,0,
            np.where(cos_hi<=-1,180,
                np.arccos(cos_hi) / pir))
    evap['hi_deg'] = hi

    # Estimate daily actual evapotranspiration (aet_d), mm
    aet_d = (24 / math.pi) * (
        sw * hi * pir +
        rx * rw * rv * (dsin(hn) - dsin(hi)) +
        (rx * rw * ru - rx * rnl) * (hn - hi) * pir
    )
    evap['aet_mm'] = aet_d

    return evap

if __name__ == "__main__":
    # Example usage
    evap = calc_daily_evap(
        lat=np.linspace(-40, 40, 40)[:,np.newaxis], 
        n=170, elv=142, y=2000, 
        sf=np.random.random((40, 80)), 
        tc=np.random.random((40, 80))*40, 
        sw=0.9)
    # print("Evaporation values:")
    # print(f"  s: {evap['s_pa.k']:.6f} Pa/K")
    # print(f"  Lv: {evap['lv_j.kg']*1e-6:.6f} MJ/kg")
    # print(f"  Patm: {evap['patm_pa']*1e-5:.6f} bar")
    # print(f"  pw: {evap['pw_kg.m3']:.6f} kg/m^3")
    # print(f"  gamma: {evap['gam_pa.k']:.6f} Pa/K")
    # print(f"  Econ: {evap['econ_m3.j']*1e9:.6f} mm^3/J")
    # print(f"  Cn: {evap['cond_mm']:.6f} mm")
    # print(f"  rx: {evap['rx']:.6f}")
    # print(f"  hi: {evap['hi_deg']:.6f} degrees")
    # print(f"  EET: {evap['eet_mm']:.6f} mm")
    print(f"  PET: {evap['pet_mm']} mm")
    # print(f"  AET: {evap['aet_mm']:.6f} mm")
