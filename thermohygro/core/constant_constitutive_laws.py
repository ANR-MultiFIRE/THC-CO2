from fenics import *
from thermohygro.core.global_constants import *
from numpy import log as np_log


# Kelvin equation to define Pc_0 by the RH and T
def Pc_Keq(RH, T):
    '''
    Kelvin Equation for setting initial capillary pressure, Pc_0 [Pa]
    Parameters
    ----------
    RH [-] : float, array_like
    A float or an array with relative humidity.

    T [K] : float, array_like
    A float or an array with temperatures.

    Returns
    -------
    Pc_Keq_vals :  float, array_like
    A flot or array object that represents the initial capillary pressure to be
    interpolated on the FEM space.

    Notes
    -----
    Can be considere as a helper function.
    '''
    np_R = float(R)
    np_M_v = float(M_v)
    Pc_Keq_vals = - np_R / np_M_v * T * float(rho_l(T)) * np_log(RH)
    return Pc_Keq_vals


# Saturation Pressure, $p_{vps}$
def Pvps(T):
    '''
    Saturation vapor pressure, Pvps [Pa]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Pvps_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation vapor pressure.

    Notes
    -----
    No notes.
    '''
    C_1 = -5800.2206
    C_2 = 1.3914993
    C_3 = -4.8640239e-2
    C_4 = 4.1764768e-5
    C_5 = -1.4452093e-8
    C_6 = 6.5459673
    Pvps_vals = exp(C_1 / T + C_2 + C_3 * T + C_4 * T**2 + C_5 * T**3 +
                    C_6 * ln(T))
    return Pvps_vals


# Derivative of the Saturation Pressure with respect to
# the Temperature, $\frac{\partial p_{vps}}{\partial T}
def dPvpsdT(T):
    '''
    Derivative of the saturation vapor pressure with respect to temperature,
    dPvpsdT [Pa/K]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvpsdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation vapor pressure with
    resoect to temperature.

    Notes
    -----
    No notes.
    '''
    C_1 = -5800.2206
    C_3 = -4.8640239e-2
    C_4 = 4.1764768e-5
    C_5 = -1.4452093e-8
    C_6 = 6.5459673
    dPvpsdT_vals = Pvps(T) * (- C_1 / T**2 + C_3 + 2 * C_4 * T +
                              3 * C_5 * T**2 + C_6 / T)
    return dPvpsdT_vals


# Water Vapour Partial Pressure
def Pv(Pg, Pc, T):
    '''
    Water vapor pressure, Pv [Pa]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Pv_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor pressure.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    Pv_vals = Pvps(T) * exp(M_v / (R * T * rho_l(T)) * (- Pc))
    return Pv_vals


# Water Vapour Density
def rho_v(Pg, Pc, T):
    '''
    Water vapor density, rho_v [kg/m^3]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_v_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor density.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    rho_v_vals = M_v / (R * T) * Pv(Pg, Pc, T)
    return rho_v_vals


# Dry Air Partial Pressure
def Pa(Pg, Pc, T):
    '''
    Air pressure, Pa [Pa]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Pa_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the air pressure.

    Notes
    -----
    If the gas pressure is lower then the vapor pressure, an air pressure of
    100 Pa is assumed.
    '''
    cond = Pg - Pv(Pg, Pc, T)
    Pa_vals = conditional(ge(cond, 100), Pg - Pv(Pg, Pc, T), 100)
    # Pa_vals = Pg - Pv(Pg, Pc, T)
    return Pa_vals


# Dry Air Density SDP_ok
def rho_a(Pg, Pc, T):
    '''
    Dry air density, rho_a [kg/m^3]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dry air density.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    rho_a_vals = (M_a / (R * T) * Pa(Pg, Pc, T))
    return rho_a_vals


# Gas Density SDP_ok
def rho_g(Pg, Pc, T):
    '''
    Gas density, rho_g [kg/m^3]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the gas density.

    Notes
    -----
    Dalton's law.
    '''
    rho_g_vals = rho_a(Pg, Pc, T) + rho_v(Pg, Pc, T)
    return rho_g_vals


# Water Density, $\rho_l$
def rho_l(T_k):
    '''
    Liquid water density, rho_l [kg/m^3]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water density.

    Notes
    -----
    After the critical temperature the liquid density is set constant.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    a_0 = 4.8863e-7
    a_1 = -1.6528e-9
    a_2 = 1.8621e-12
    a_3 = 2.4266e-13
    a_4 = -1.5996e-15
    a_5 = 3.3703e-18
    b_0 = 1021.3e0
    b_1 = -7.7377e-1
    b_2 = 8.7696e-3
    b_3 = -9.2118e-5
    b_4 = 3.3534e-7
    b_5 = -4.4034e-10
    p_l_1 = 1.0e7
    p_l_ref = 2.0e7
    cond_1 = (b_0 + b_1 * T + b_2 * T**2 + b_3 * T**3 + b_4 * T**4 +
              b_5 * T**5) + ((p_l_1 - p_l_ref) * (a_0 + a_1 * T + a_2 * T**2 +
                                                  a_3 * T**3 + a_4 * T**4 +
                                                  a_5 * T**5))
    cond_2 = ((b_0 + b_1 * T_cr_C + b_2 * T_cr_C**2 + b_3 * T_cr_C**3 +
               b_4 * T_cr_C**4 + b_5 * T_cr_C**5) +
              ((p_l_1 - p_l_ref) * (a_0 + a_1 * T_cr_C + a_2 * T_cr_C**2 +
                                    a_3 * T_cr_C**3 + a_4 * T_cr_C**4 +
                                    a_5 * T_cr_C**5)))
    rho_l_vals = conditional(lt(T, T_cr_C), cond_1, cond_2)
    return rho_l_vals


# Derivative of Water Density, $\frac{\partial \rho_l}{\partial T}$ SDP_ok
def drho_ldT(T_k):
    '''
    Derivative of liquid water density, drho_ldT [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_ldT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of liquid water density with respect to
    temperature.

    Notes
    -----
    After the critical temperature the liquid density is set constant.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    a_1 = -1.6528e-9
    a_2 = 1.8621e-12
    a_3 = 2.4266e-13
    a_4 = -1.5996e-15
    a_5 = 3.3703e-18
    b_1 = -7.7377e-1
    b_2 = 8.7696e-3
    b_3 = -9.2118e-5
    b_4 = 3.3534e-7
    b_5 = -4.4034e-10
    p_l_1 = 1.0e7
    p_l_ref = 2.0e7
    cond_1 = ((b_1 + 2 * b_2 * T + 3 * b_3 * T**2 + 4 * b_4 * T**3 +
               5 * b_5 * T**4) + (p_l_1 - p_l_ref) *
              (a_1 + 2 * a_2 * T + 3 * a_3 * T**2 + 4 * a_4 * T**3 +
               5 * a_5 * T**4))
    cond_2 = ((b_1 + 2 * b_2 * T_cr_C + 3 * b_3 * T_cr_C**2 +
               4 * b_4 * T_cr_C**3 + 5 * b_5 * T_cr_C**4) + (p_l_1 - p_l_ref) *
              (a_1 + 2 * a_2 * T_cr_C + 3 * a_3 * T_cr_C**2 +
               4 * a_4 * T_cr_C**3 + 5 * a_5 * T_cr_C**4))
    drho_ldT_vals = conditional(lt(T, T_cr_C), cond_1, cond_2)
    return drho_ldT_vals


# Viscosity of Water Vapour, $\mu_v$ SDP_ok
def mu_v(T):
    '''
    Viscosity of water vapor, mu_v [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_v_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor's viscosity.

    Notes
    -----
    No notes.
    '''
    mu_v_0 = 8.85e-6
    alpha_v = 3.53e-8
    mu_vals = mu_v_0 + alpha_v * (T - T_0_K)
    return mu_vals


# Viscosity of Dry Air, $\mu_a$ SDP_ok
def mu_a(T):
    '''
    Viscosity of dry air, mu_a [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dry air viscosity.

    Notes
    -----
    No notes.
    '''
    mu_a_0 = 17.17e-6
    alpha_a = 4.733e-8
    beta_a = 2.222e-11
    mu_a_vals = mu_a_0 + alpha_a * (T - T_0_K) + beta_a * (T - T_0_K)**2
    return mu_a_vals


# Viscosity of Gas, $\mu_g$
def mu_g(Pg, Pc, T):
    '''
    Viscosity of gas, mu_g [Pa s]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the overall gas air viscosity.

    Notes
    -----
    No notes.
    '''
    mu_g_vals = conditional(ge(Pg - Pv(Pg, Pc, T), 0),
                            (mu_v(T) + (mu_a(T) - mu_v(T)) *
                             (1 - (Pv(Pg, Pc, T) / Pg))**0.6083),
                            mu_v(T))
    return mu_g_vals

# Viscosity of Liquid Water, $\mu_l$ SDP_ok
def mu_l(T):
    '''
    Viscosity of liquid water, mu_l [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water viscosity.

    Notes
    -----
    No notes.
    '''
    mu_l_vals = 0.6612 * (T - 229)**(-1.562)
    return mu_l_vals


# Enthalpy of Evaporation, $\Delta H_{vp}$
def H_vap(T):
    '''
    Enthalpy of vaporization of liquid water, H_vap [J/kg]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    H_vap_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents enthalpy of vaporization of liquid water.

    Notes
    -----
    No notes.
    '''
    return conditional(lt(T, T_cr), 2.6725e5 * ((T_cr - T)**0.38), 0.0)


def dH_vapdT(T):
    '''
    Derivative of the enthalpy of vaporization of liquid water with respect to
    temperature, dH_vapdT [J/(kg K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dH_vapdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents enthalpy of vaporization of liquid water.

    Notes
    -----
    No notes.
    '''
    dH_vapdT_vals = conditional(lt(T, T_cr),
                                0.38 * 2.6725e5 * (T_cr - T)**(0.38 - 1),
                                0.0)
    return dH_vapdT_vals


def M_g(Pg, Pc, T):
    '''
    Molar Mass of the gas mixture, M_g [kg/mol]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    M_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents molar mass of the gas mixture.

    Notes
    -----
    No notes.
    '''
    cond_1 = Pg - Pv(Pg, Pc, T)
    M_g_vals_1 = M_a + (M_v - M_a) * Pv(Pg, Pc, T) / Pg
    M_g_vals = conditional(ge(cond_1, 0), M_g_vals_1, M_v)
    return M_g_vals


def Cp_g(Pg, Pc, T):
    '''
    Specific heat of gas mixture, Cp_g [kg/mol]
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Cp_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the specific heat of the gas mixture.

    Notes
    -----
    No notes.
    '''
    Cp_g_vals = (rho_g(Pg, Pc, T) *  Cp_a + rho_v(Pg, Pc, T) * (Cp_v - Cp_a))
    return Cp_g_vals


# Partial Drivatives of Water Vapour Density
def drho_vdPc(Pg, Pc, T):
    '''
    Derivative of the water vapor's density with respect to capillary pressure,
    drho_vdPc [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    drho_vdPc_vals = M_v / (R * T) * dPvdPc(Pg, Pc, T)
    return drho_vdPc_vals


def dPvdPc(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to capillary pressure,
    drho_vdPc [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    dPvdPc_vals = - rho_v(Pg, Pc, T) / rho_l(T)
    return dPvdPc_vals


def drho_vdPg(Pg, Pc, T):
    '''
    Derivative of the water vapor's density w/ respect to gas pressure,
    drho_vdPg [kg/(m^3 Pa)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    drho_vdPg_vals = M_v / (R * T) * dPvdPg(Pg, Pc, T)
    return drho_vdPg_vals


def dPvdPg(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to gas pressure,
    dPvdPg [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    dPvdPg_vals = rho_v(Pg, Pc, T) / rho_l(T)
    return dPvdPg_vals


def drho_vdT(Pg, Pc, T):
    '''
    Derivative of the water vapor's density w/ respect to gas pressure,
    dPvdPg [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    return (M_v / (R * T)) * dPvdT(Pg, Pc, T) - rho_v(Pg, Pc, T) / T


def dPvdT(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to temperature,
    dPvdT [Pa/K]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    dPvdT_vals = ((Pv(Pg, Pc, T) / Pvps(T)) * dPvpsdT(T) -
                  rho_v(Pg, Pc, T) / rho_l(T) * (- Pc ) *
                  (1 / rho_l(T) * drho_ldT(T) + 1 / (T)))
    return dPvdT_vals


# Partial Drivatives of Dry Air Density
def drho_adPc(Pg, Pc, T):
    '''
    Derivative of the dry air's density with respect to capillary pressure,
    drho_adPc [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    drho_adPc_vals = M_a / (R * T) * dPadPc(Pg, Pc, T)
    return drho_adPc_vals


def dPadPc(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to capillary pressure,
    dPadPc [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    dPadPc = rho_v(Pg, Pc, T) / rho_l(T)
    return dPadPc


def drho_adPg(Pg, Pc, T):
    '''
    Derivative of the dry air's density w/ respect to gas pressure,
    drho_adPg [kg/(m^3 Pa)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    drho_adPg = M_a / (R * T) * dPadPg(Pg, Pc, T)
    return drho_adPg


def dPadPg(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to temperature,
    dPadPg [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    dPadPg = 1 - rho_v(Pg, Pc, T) / rho_l(T)
    return dPadPg


def drho_adT(Pg, Pc, T):
    '''
    Derivative of the dry air's density w/ respect to temperature,
    drho_adT [kg/(m^3K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    drho_adT = M_a / (R * T) * dPadT(Pg, Pc, T) - rho_a(Pg, Pc, T) / T
    return drho_adT


def dPadT(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to temperature,
    dPadT [Pa/K]
    
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    dPadT_vals = - dPvdT(Pg, Pc, T)
    return dPadT_vals

def K_h(T):
    '''
    Henry constant for setting influence of temperature on CO2 diffusivity, K_h [mol/m3/Pa]
    
    Parameters
    ----------
    T [K] : float, array_like
    A float or an array with temperatures.

    Returns
    -------
    K_h_vals :  float, array_like
    Henry constant for setting influence of temperature on CO2 diffusivity, K_h [mol/m3/Pa]

    Notes
    -----
    Can be considere as a helper function.
    '''
    H_0 = 3.36e-4
    K_h_vals =  H_0 *  exp(- delta_H_R * (1/T - 1/T_ref_1)) 
    return K_h_vals


def rho_co2(T, Pco2):
    '''
    Molar volume of the CO2 gas, rho_co2 [kg/m3]
    
    Parameters
    ----------

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Pco2 [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    rho_co2_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents Molar volume of the CO2 gas.

    Notes
    -----
    No notes.
    '''
    rho_co2_vals = M_co2 / (R * T) * Pco2
    return rho_co2_vals

def drho_co2dPco2(T, Pco2):
    '''
    Derivative of the co2 gas molar volume with respect to CO2 pressure,
    drho_co2dPco2 [mol/m3/Pa]
    
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Pco2 [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    drho_co2dPco2_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the co2 gas molar volume with respect 
    to CO2 pressure, drho_co2dPco2 [mol/m3/Pa]

    Notes
    -----
    No notes.
    '''
    drho_co2dPco2_vals = M_co2 / (R * T) 
    return drho_co2dPco2_vals

def drho_co2dT(T, Pco2):
    '''
    Derivative of the co2 gas molar volume with respect to temperature T,
    drho_co2dT [mol/m3/K]
    
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Pco2 [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    drho_co2dT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the co2 gas molar volume with respect 
    to temperature, drho_co2dPco2 [mol/m3/K]

    Notes
    -----
    No notes.
    '''
    drho_co2dT_vals = - M_co2 / (R * (T**2)) * Pco2 
    return drho_co2dT_vals

def to_ch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3):
    '''
    characteristic reaction time of portlandite in seconds
    
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Pco2 [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    nch[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    ncsh[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    ncaco3[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    to_ch_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents  characteristic reaction time of portlandite to_ch [s]

    Notes
    -----
    No notes.
    '''
    to_ch_1 = 3 * nch * vch / rch_0 * ((nch/nch_0)**(2/3)) *  D_caco3    
    to_ch_2 = lam + rch_0 * (((1+(1-nch/nch_0)*(vcc/vch-1))**(1/3))-(nch/nch_0)**(1/3))
    to_ch_vals = to_ch_1 / to_ch_2
    return to_ch_vals

def to_csh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3):
    '''
    characteristic reaction time of CSH in seconds
    
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Pco2 [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    nch[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    ncsh[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    ncaco3[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    to_csh_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents  characteristic reaction time of CSH to_csh [s]

    Notes
    -----
    No notes.
    '''
    to_csh_1 = n_0 * ncsh
    to_csh_2 = alpha_csh * ncsh_0
    to_csh_vals = to_csh_1 / to_csh_2
    return to_csh_vals
