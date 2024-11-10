import numpy as np
from numpy.typing import ArrayLike

from utils import dsin, dcos, julian_day

def berger_tls(n: ArrayLike, N: ArrayLike, ke=0.01670, keps=23.44, komega=283, pir=np.pi / 180):
    # Variable substitutes:
    xee = ke ** 2
    xec = ke ** 3
    xse = np.sqrt(1 - ke ** 2)

    # Mean longitude for vernal equinox:
    xlam = (ke / 2.0 + xec / 8.0) * (1 + xse) * np.sin(komega * pir) - \
           xee / 4.0 * (0.5 + xse) * np.sin(2.0 * komega * pir) + \
           xec / 8.0 * (1.0 / 3.0 + xse) * np.sin(3.0 * komega * pir)
    xlam = 2.0 * xlam / pir

    # Mean longitude for day of year:
    dlamm = xlam + (n - 80.0) * (360.0 / N)

    # Mean anomaly:
    anm = dlamm - komega
    ranm = anm * pir

    # True anomaly (uncorrected):
    ranv = ranm + (2.0 * ke - xec / 4.0) * np.sin(ranm) + \
           5.0 / 4.0 * xee * np.sin(2.0 * ranm) + \
           13.0 / 12.0 * xec * np.sin(3.0 * ranm)
    anv = ranv / pir

    # True longitude:
    my_tls = anv + komega
    my_tls = np.where(my_tls<0, my_tls+360, my_tls)
    my_tls = np.where(my_tls>360, my_tls-360, my_tls)

    # True anomaly:
    my_nu = my_tls - komega
    my_nu = np.where(my_nu<0, my_nu+360, my_nu)
    return my_nu, my_tls

def calc_daily_solar(
        lat: ArrayLike, n, 
        elv=0, y=0, sf: float|ArrayLike=1, tc=23.0, ke=0.01670, keps=23.44, komega=283, kA=107,
        kalb_sw=0.17, kalb_vis=0.03, kb=0.20, kc=0.25, kd=0.50, kfFEC=2.04, kGsc=1360.8):
    """This function calculates daily solar radiation fluxes.

    Args:
        lat (ArrayLike): latitudes, decimal degrees
        n (float): day of year, e.g. `n=15` for 15 Jan
        elv (float, optional): elevation, m A.S.L. Defaults to 0.
        y (int, optional): year. Defaults to 0.
        sf (float|ArrayLike, optional): fraction of sunshine hours. Defaults to 1.
        tc (ArrayLike, optional): mean daily air temperature. Defaults to 23.0.
        ke (float, optional): eccentricity of earth's orbit. Defaults to 0.01670.
        keps (float, optional): obliquity of earth's elliptic. Defaults to 23.44.
        komega (int, optional): lon. of perihelion, degrees. Defaults to 283.
        kA (int, optional): empirical constant, degrees Celsius. Defaults to 107.
        kalb_sw (float, optional): shortwave albedo. Defaults to 0.17.
        kalb_vis (float, optional): visible light albedo. Defaults to 0.03.
        kb (float, optional): empirical constant. Defaults to 0.20.
        kc (float, optional): cloudy transmittivity. Defaults to 0.25.
        kd (float, optional): angular coefficient of transmittivity. Defaults to 0.50.
        kfFEC (float, optional): flux-to-energy conversion. Defaults to 2.04.
        kGsc (float, optional): solar constant, W/m^2. Defaults to 1360.8.

    Returns:
        _type_: _description_
    """    
    pir = np.pi / 180
    # 初始化返回字典
    solar = {}

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 01. 计算年度天数 kN ~~~~~~~~~~~~~~~~~~~~~~~ #
    if y == 0:
        kN = 365
    else:
        kN = julian_day(y + 1, 1, 1) - julian_day(y, 1, 1)
    solar['kN'] = kN

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 02. 计算日心角（nu 和 lambda） ~~~~~~~~~~~~~~~~~~~~~~~ #
    nu, lam = berger_tls(n, kN, ke, keps, komega)
    solar['nu_deg'] = nu
    solar['lambda_deg'] = lam

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 03. 计算地球与太阳的距离因子 dr ~~~~~~~~~~~~~~~~~~~~~~~ #
    kee = ke ** 2
    rho = (1 - kee) / (1 + ke * dcos(nu))
    dr = (1 / rho) ** 2
    solar['rho'] = rho
    solar['dr'] = dr

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 04. 计算赤纬角（delta） ~~~~~~~~~~~~~~~~~~~~~~~ #
    delta = np.arcsin(dsin(lam) * dsin(keps))
    delta = np.degrees(delta)
    solar['delta_deg'] = delta

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 05. 计算变量替代物 ru 和 rv ~~~~~~~~~~~~~~~~~~~~~~~ #
    ru = dsin(delta) * dsin(lat)
    rv = dcos(delta) * dcos(lat)
    solar['ru'] = ru
    solar['rv'] = rv

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 06. 计算日落时角（hs） ~~~~~~~~~~~~~~~~~~~~~~~ #
    r = ru/rv
    hs = np.where(r >= 1.0, 
            180, 
            np.where(r <= -1.0,
                0,
                np.degrees(np.arccos(-1.0 * r))))
    solar['hs_deg'] = hs

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 07. 计算外辐射（ra_d） ~~~~~~~~~~~~~~~~~~~~~~~ #
    ra_d = (86400 / np.pi) * kGsc * dr * (ru * np.radians(hs) + rv * dsin(hs))
    solar['ra_j.m2'] = ra_d

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 08. 计算透射率（tau） ~~~~~~~~~~~~~~~~~~~~~~~ #
    tau_o = (kc + kd * sf)
    tau = tau_o * (1 + (2.67e-5) * elv)
    solar['tau_o'] = tau_o
    solar['tau'] = tau

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 09. 计算光合有效辐射（ppfd_d） ~~~~~~~~~~~~~~~~~~~~~~~ #
    ppfd_d = (1e-6) * kfFEC * (1 - kalb_vis) * tau * ra_d
    solar['ppfd_mol.m2'] = ppfd_d

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 10. 估算净长波辐射（rnl） ~~~~~~~~~~~~~~~~~~~~~~~ #
    rnl = (kb + (1 - kb) * sf) * (kA - tc)
    solar['rnl_w.m2'] = rnl

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 11. 计算辐射变数 rw ~~~~~~~~~~~~~~~~~~~~~~~ #
    rw = (1 - kalb_sw) * tau * kGsc * dr
    solar['rw'] = rw

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 12. 计算净辐射交叉角（hn） ~~~~~~~~~~~~~~~~~~~~~~~ #
    net_rad = (rnl - rw * ru) / (rw * rv)
    hn = np.where(net_rad>=1, 0, 
            np.where(net_rad<=1, 180, 
                np.degrees(np.arccos(net_rad))))
    solar['hn_deg'] = hn

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 13. 计算白天净辐射（rn_d） ~~~~~~~~~~~~~~~~~~~~~~~ #
    rn_d = (86400 / np.pi) * (hn * pir * (rw * ru - rnl) + rw * rv * dsin(hn))
    solar['rn_j.m2'] = rn_d

    # ~~~~~~~~~~~~~~~~~~~~~~~~ 14. 计算夜间净辐射（rnn_d） ~~~~~~~~~~~~~~~~~~~~~~~ #
    rnn_d = (86400 / np.pi) * (rw * rv * (dsin(hs) - dsin(hn)) +
                                 rw * ru * (hs - hn) * pir -
                                 rnl * (np.pi - hn * pir))
    solar['rnn_j.m2'] = rnn_d

    return solar

if __name__=='__main__':
    import time
    lat = np.linspace(-40, 40, 40)[:,np.newaxis]
    n = 10
    sf = np.random.random((40, 800))
    tick = time.time()
    calc_daily_solar(lat=lat, n=n, sf=sf)
    tock = time.time()
    duration = tock - tick
    print(f'running {duration}')