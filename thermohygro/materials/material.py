from fenics import *
from thermohygro.core.global_constants import *
from thermohygro.core.constant_constitutive_laws import *
import numpy as np

H_dehyd          = Constant(2.5e6)           # Enthalpie of Dehydration
Q_2 = 7e6
Q_3 = 18.6e6
Q_3_carb = 4.6e6
N = 10
E_0 = 1
z = 5
b = 2.65
b_carb = 3.7

A_KRL = 0.6                         # Coef. of Liquid Relative Permeability
A_KRG = 0.6                         # Coef. of Gas Relative Permeability
A_KRG_carb = 0.6
# Initial Values
K_0 = 12.2e-21                     # Initial Permeability
rho_s_0 = 1920                      # Initial Solid's Density
A_rho_s = 0.2235
# phi_0 = 0.1368                      # Initial Porosity
# A_phi = 0.72e-3                     # Porosity Coefficient
phi_0 = 0.4                      # Initial Porosity
A_phi = 0.78e-5                     # Porosity Coefficient
lambda_d_0 = 1.67                   # Initial Thermal Conductivity
Cp_s_0 = 948                        # Initial Specific Heat

def rho_s(T):
    '''
    Solid's density, rho_s [kg/m^3]
    RHO_S_OPTION == 1

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid's density.

    Notes
    -----
    There are currently 2 options for laws. RHO_S_OPTION == 1 is a
    constant function for density, while RHO_S_OPTION == 2 is a linear
    function of temperature. Such options should be defined within
    'materials_input.py'.
    '''

    rho_s_vals = rho_s_0 + A_rho_s * (T - T_ref_1)
    return rho_s_vals

def drho_sdT(T):
    '''
    Derivative of solid's density w/ respect to T, drho_sdT [kg/(m^3 . K)]
    RHO_S_OPTION == 2

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_sdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the solid's density with respect to
    T.

    Notes
    -----
    There are currently 2 options for laws. RHO_S_OPTION == 1 is a
    constant function for density, while RHO_S_OPTION == 2 is a linear
    function of temperature. Such options should be defined within
    'materials_input.py'.
    '''
    return A_rho_s


# Total Porosity, $\phi$ and total Porosity derivative with respect to
# Temperature, $frac{d \phi}{dT}$. The PHI_OPTION may or may not be defined in
# 'materials_input.py'.
def phi(T):
    '''
    Solid's porosity, phi [-]
    PHI_OPTION == 1

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    phi_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid's porosity.

    Notes
    -----
    There are currently 2 options for laws. PHI_OPTION == 1 is a
    linear function of temperature, while PHI_OPTION == 2 is a power
    law of temperature. Such options should be defined within
    'materials_input.py'. ATTENTION: WHEN PHI_OPTION IS NOT EXPLICITLY
    DEFINED WITHIN 'materials_input.py', THE DEFAULT BEHAVIOR IS THE
    LINEAR FUNCTION (PHI_OPTION == 1).
    '''
    phi_vals = phi_0 + A_phi * (T - T_ref_1)
    return phi_vals

# Total Porosity Derivative with respect to Temperature
def dphidT(T):
    '''
    Derivative of solid's porosity with respect to T, dphidT [1/K]
    PHI_OPTION == 1

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dphidT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the solid's porosity with respect
    to T.

    Notes
    -----
    There are currently 2 options for laws. PHI_OPTION == 1 is a
    linear function of temperature, while PHI_OPTION == 2 is a power
    law of temperature. Such options should be defined within
    'materials_input.py'. ATTENTION: WHEN PHI_OPTION IS NOT EXPLICITLY
    DEFINED WITHIN 'materials_input.py', THE DEFAULT BEHAVIOR IS THE
    LINEAR FUNCTION (PHI_OPTION == 1).
    '''
    dphidT_vals = A_phi
    return dphidT_vals


# Intrinsic Permeabiliy, $K$. The K_OPTION may or may not be defined within
# 'materials_input.py'.
def K(T,phi,ncaco3):
    '''
    Intrinsic permeability, K [m^2]
    K_OPTION == 1

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    K_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the intrisic permeability.

    Notes
    -----
    There are currently 2 options for laws. K_OPTION == 1 is an
    exponential function of temperature and gas pressure,
    while PHI_OPTION == 2 is a power law of temperature only.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN K_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE
    EXPONENTIAL FUNCTION OF TEMPERATURE AND GAS PRESSURE
    (K_OPTION == 1).
    '''
    T_dr_1 = 20 + 273.15
    T_dr_2 = 35.7 
    K_vals =  K_0 * exp(exp((T - T_dr_1)/ T_dr_2) - 1) * (phi/phi_0)**3 * ((1-phi_0) / (1 - phi))**2
    return K_vals

def k_rl(Pg, Pc, T, ncaco3):
    '''
    Liquid water relative permeability, k_rl [-]
    KRL_OPTION == 1

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
    k_rl_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water relative permeability.

    Notes
    -----
    There are currently 3 options for laws. KRL_OPTION = 1, 2 or 3.
    '''
    A = A_KRG
    A_carb = A_KRG_carb
    k_rl_vals = S_l(Pg, Pc, T, ncaco3)**0.5 * (1 - (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A))**A)**2
    #k_rl_vals = ((S_l(Pg, Pc, T, ncaco3)**0.5 * (1 - (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A))**A)**2) *(1-ncaco3/ncaco3_max)) + ((S_l(Pg, Pc, T, ncaco3)**0.5 * (1 - (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A_carb))**A_carb)**2) *(1-ncaco3/ncaco3_max))
    return k_rl_vals


# Gas Relative Permeability, $k_{rg}$.
def k_rg(Pg, Pc, T, ncaco3):
    '''
    Gas relative permeability, k_rg [-]
    KRG_OPTION == 1

    Parameters
    ----------
    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    k_rg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the gas relative permeability.

    Notes
    -----
    There are currently 3 options for laws. KRG_OPTION = 1, 2 or 3.
    '''
    A = A_KRG
    A_carb = A_KRG_carb
    k_rg_vals = (1 - S_l(Pg, Pc, T, ncaco3))**0.5 * (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A))**(2 * A)
    #k_rg_vals = ((1 - S_l(Pg, Pc, T, ncaco3))**0.5 * (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A))**(2 * A) *(1-ncaco3/ncaco3_max)) + ((1 - S_l(Pg, Pc, T, ncaco3))**0.5 * (1 - S_l(Pg, Pc, T, ncaco3)**(1 / A_carb))**(2 * A_carb) *(ncaco3/ncaco3_max))
    return k_rg_vals


# Mass Diffusivity , $D_{eff}$.
def D_va(Pg, T):
    '''
    Diffusivity of vapor in air, D_va [m^2/s]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    D_va_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the diffusivity of vapor in air.

    Notes
    -----
    Other diffusivity laws may be added.
    '''
    A_v = 1.667
    D_va_vals = D_v_0 * (T / T_0_K)**A_v * (Pg_ref / Pg)
    return D_va_vals


def f_s(Pg, Pc, T, phi, ncaco3):
    '''
    Turtuosity and the available pore space for gas, f_s [-]

    Parameters
    ----------
    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    f_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the turtuosity and available pore space for gas.

    Notes
    -----
    It is used to derive the effective diffusivity, D_eff.
    '''

    tau = phi**(1 / 3) * (1 - S_l(Pg, Pc, T, ncaco3))**(7 / 3)
    pore_space_gas = phi * (1 - S_l(Pg, Pc, T, ncaco3))
    f_s_vals = tau * pore_space_gas
    return f_s_vals


def D_eff(Pg, Pc, T, phi, ncaco3):
    '''
    Effective diffusivity of vapor in air, D_eff [m^2/s]

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
    D_eff_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective diffusivity of vapor in air.

    Notes
    -----
    Other diffusivity laws may be added.
    '''
    D_eff_vals = f_s(Pg, Pc, T, phi, ncaco3) * D_va(Pg, T)
    return D_eff_vals


# Thermal Conductivity, $\lambda$. The LAMB_OPTION may or may not be defined
# within 'materials_input.py'.
def lambda_d(T):
    '''
    Thermal conductiity of dry concrete, lambda_d [W/(m K)]
    LAMB_OPTION == 1

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    lambda_d_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the thermal conductivity of dry concrete.

    Notes
    -----
    There are currently 2 options for laws. LAMB_OPTION == 1 is a
    linear function of temperature for the dry material and that also
    accounts for the existance of water, while LAMB_OPTION == 2 is
    a power law of temperature only.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN LAMB_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE
    LINEAR FUNCTION OF TEMPERATURE AND THE CONSIDERATION OF LIQUID
    (LAMB_OPTION == 1).
    '''
    A_lambda = - 0.0005
    lambda_d_vals = lambda_d_0 * (1 + A_lambda * (T - T_ref_1))
    return lambda_d_vals

def lambda_eff(Pg, Pc, T, phi, ncaco3):
    '''
    Effective thermal conductivity of moist concrete,
    lambda_eff [W/(m . K)]
    LAMB_OPTION == 1

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function
    for defining variational formulations for FEniCS FEM library.

    Returns
    -------
    lambda_eff_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective thermal conductivity of moist
    concrete.

    Notes
    -----
    There are currently 2 options for laws. LAMB_OPTION == 1 is a
    linear function of temperature for the dry material and that also
    accounts for the existance of water, while LAMB_OPTION == 2 is
    a power law of temperature only.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN LAMB_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE
    LINEAR FUNCTION OF TEMPERATURE AND THE CONSIDERATION OF LIQUID
    (LAMB_OPTION == 1).
    '''
    # lambda_eff = lambda_d(T)
    lambda_eff =   lambda_d(T) * (1 + 4 * (S_l(Pg, Pc, T, ncaco3) * phi *
                                         rho_l(T)) /
                                ((1 - phi) * rho_s(T)))
    return lambda_eff


# Specific Heat of Solid Concrete, $C_ps$
def Cp_s(T):
    '''
    Specific heat of solid concrete, Cp_s [J/(kg . K)]

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Cp_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid concrete specific heat.

    Notes
    -----
    ATTENTION: THE LAW IS DEFINED IN CELSIUS, T [K] IS THUS CONVERTED TO
    T_C [C].
    '''
    T_C = T - T_0_K
    A_c = 0.35
    Cp_s_vals = Cp_s_0 + A_c * T_C
    return Cp_s_vals


# Effective Thermal Capacity, $\rho C_p$
def rhoCp(Pg, Pc, T, phi, ncaco3):
    '''
    Effective thermal capacity of concrete, rhoCp [J/(K . m^3)]

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
    rho_Cp_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective thermal capacity of concrete.

    Notes
    -----
    Cp_g might vary in other implementions.
    '''
    Cp_g = (rho_g(Pg, Pc, T) * Cp_a + rho_v(Pg, Pc, T) * (Cp_v - Cp_a))
    rhoCp_s = (1 - phi) * rho_s(T) * Cp_s(T)
    rhoCp_l = phi * S_l(Pg, Pc, T, ncaco3) * rho_l(T) * Cp_l
    rhoCp_g = phi * (1 - S_l(Pg, Pc, T, ncaco3)) * Cp_g
    rho_Cp_vals = rhoCp_s + rhoCp_l + rhoCp_g
    return rho_Cp_vals


# Mass of Dehydration, $\Delta m_{des}$ and its derivative with respect
# to $T$, $\frac{\partial \Delta m_{des}}{\partial T}$. The LAMB_OPTION may or
# may not be defined within 'materials_input.py'.
def f_d(T_k):
    '''
    Dehydration dependence on the Temperature, f_d [K]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    f_d_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dehydration dependence on the Temperature.

    Notes
    -----
    There are currently 2 options for laws. DEHYDE_OPTION == 1 is has
    its argument divided by 2, while DEHYDE_OPTION == 2 is not divided
    by two.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN DEHYDE_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE ARGUMENT DIVIDED
    BY TWO.
    '''
    T = T_k - T_0_K
    cond = ((1 + sin(pi / 2 * (1 - 2 * exp(-0.004 * (T - 105))))) / 2)
    f_d_vals = conditional(lt(T, 105), 0, cond)
    return f_d_vals

def m_dehyd(T):
    '''
    Dehydration mass, m_dehyd [kg/m^3]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    m_dehyd_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dehydration mass release.

    Notes
    -----
    There are currently 2 options for laws. DEHYDE_OPTION == 1 is has
    its argument divided by 2, while DEHYDE_OPTION == 2 is not divided
    by two.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN DEHYDE_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE ARGUMENT DIVIDED
    BY TWO.
    '''
    f_c = 200
    f_s = 0.24
    f_m = 0.4
    m_dehyd_vals = f_c * f_s * f_m * f_d(T)
    return m_dehyd_vals

def df_ddT(T_k):
    '''
    Derivative of the dehydration dependence w/ respect to T,
    df_ddT [kg/(m^3 K)]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    df_ddT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the dehydration mass release
    dependence with temperature.

    Notes
    -----
    There are currently 2 options for laws. DEHYDE_OPTION == 1 is has
    its argument divided by 2, while DEHYDE_OPTION == 2 is not divided
    by two.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN DEHYDE_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE ARGUMENT DIVIDED
    BY TWO.
    '''
    T = T_k - T_0_K
    cond = (cos(pi / 2 * (1 - 2 * exp(-0.004 * (T - 105)))) *
            pi * 0.004 / 2 * exp(-0.004 * (T - 105)))
    df_ddT_vals = conditional(lt(T, 105), 0, cond)
    return df_ddT_vals

def dm_dehyddT(T):
    '''
    Derivative of the dehydration mass with respect to T,
    dm_dehyddT [kg/(m^3 K)]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dm_dehyddT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the mass release of
    dehydration with respect to temperetature.

    Notes
    -----
    There are currently 2 options for laws. DEHYDE_OPTION == 1 is has
    its argument divided by 2, while DEHYDE_OPTION == 2 is not divided
    by two.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN DEHYDE_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE ARGUMENT DIVIDED
    BY TWO.
    '''
    f_c = 200
    f_s = 0.24
    f_m = 0.4
    dm_dehyddT_vals = f_c * f_s * f_m * df_ddT(T)
    return dm_dehyddT_vals


# Saturation as a function of $T$ and $P_c$, $S_l(T, P_c)$ # and its derivative
# with respect to $T$, $\frac{\partial S_l}{\partial T}$

def gamma(T):
    A = 5.11646108e-16
    B = 8.42927124e-13
    C = 5.61016172e-10
    D = 1.90043609e-07
    E = 3.32249965e-05
    F = 2.68991607e-03
    G = 1.50691817e-01
    gamma_vals = (A*T**6 - B*T**5 + C*T**4 - D*T**3 + E*T**2 - F*T + G)
    return gamma_vals

def a(T_k):
    '''
    Parameter a of second option of saturation as a function of temperature
    and capillary pressure, a [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_b_C = 100
    T_cr_C = T_cr - T_0_K
    Q_0 = ((Q_3 - Q_2) * (1 + 2 * ((T - T_b_C) / (T_cr_C - T_b_C))**3 -
                          3 * ((T - T_b_C) / (T_cr_C - T_b_C))**2))
    cond_1 = (Q_3)
    cond_2 = (Q_0 + Q_2)
    a_vals = conditional(lt(T, T_b_C), cond_1, cond_2)
    return a_vals


def a_carb(T_k):
    '''
    Parameter a of second option of saturation as a function of temperature
    and capillary pressure, a [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_b_C = 100
    T_cr_C = T_cr - T_0_K
    Q_0 = ((Q_3 - Q_2) * (1 + 2 * ((T - T_b_C) / (T_cr_C - T_b_C))**3 -
                          3 * ((T - T_b_C) / (T_cr_C - T_b_C))**2))
    cond_1 = (Q_3_carb)
    cond_2 = (Q_0 + Q_2)
    a_carb_vals = conditional(lt(T, T_b_C), cond_1, cond_2)
    return a_carb_vals
def E(T_k):
    '''
    Parameter E of second option of saturation as a function of temperature
    and capillary pressure, E [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    E_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    T_ref_1_C = T_ref_1 - T_0_K
    cond_1 = (((T_cr_C - T_ref_1_C) / (T_cr_C - T))**N)
    cond_2 = ((N / z * E_0 * T) + (E_0 - N / z * E_0 * (T_cr_C - z)))
    E_vals = conditional(lt(T, T_cr_C), cond_1, cond_2)
    return E_vals

def dm_dehyddT(T):
    '''
    Derivative of the dehydration mass with respect to T,
    dm_dehyddT [kg/(m^3 K)]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dm_dehyddT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the mass release of
    dehydration with respect to temperetature.

    Notes
    -----
    '''
    f_c = 200
    f_s = 0.24
    f_m = 0.4
    dm_dehyddT_vals = f_c * f_s * f_m * df_ddT(T)
    return dm_dehyddT_vals

def df_ddT(T_k):
    '''
    Derivative of the dehydration dependence w/ respect to T,
    df_ddT [kg/(m^3 K)]
    DEHYDE_OPTION == 1
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    df_ddT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the dehydration mass release
    dependence with temperature.

    Notes
    -----
    There are currently 2 options for laws. DEHYDE_OPTION == 1 is has
    its argument divided by 2, while DEHYDE_OPTION == 2 is not divided
    by two.
    Such options should be defined within 'materials_input.py'.
    ATTENTION: WHEN DEHYDE_OPTION IS NOT EXPLICITLY DEFINED WITHIN
    'materials_input.py', THE DEFAULT BEHAVIOR IS THE ARGUMENT DIVIDED
    BY TWO.
    '''
    T = T_k - T_0_K
    cond = (cos(pi / 2 * (1 - 2 * exp(-0.004 * (T - 105)))) *
            pi * 0.004 / 2 * exp(-0.004 * (T - 105)))
    df_ddT_vals = conditional(lt(T, 105), 0, cond)
    return df_ddT_vals

def G(T, Pc):
    '''
    Parameter G of second option of saturation as a function of temperature
    and capillary pressure, G [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    G_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    G_vals = (E(T) / a(T) * Pc)**(b / (b - 1))
    return G_vals

def G_carb(T, Pc):
    '''
    Parameter G of second option of saturation as a function of temperature
    and capillary pressure, G [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    G_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    G_carb_vals = (E(T) / a_carb(T) * Pc)**(b_carb / (b_carb - 1))
    return G_carb_vals

def S_l(Pg, Pc, T, ncaco3):
    '''
    Saturation of liquid water before carbonation as a function of temperature and
    capillary pressure, S_l [-]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    ncaco3[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    S_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    S_l_vals = conditional(lt(T, T_cr), ((G(T, Pc) + 1)**(-1 / b) * (1-ncaco3 / ncaco3_max)) 
                           + ((G_carb(T, Pc) + 1)**(-1 / b_carb) * (ncaco3 / ncaco3_max)), 0)
    return S_l_vals

def dadT(T_k):
    '''
    Derviative of the parameter a with respect to
    temperature, dadT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dadT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter a with respect to
    temperature.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_b_C = 100
    T_cr_C = T_cr - T_0_K
    cond_1 = 0
    cond_2 = (6 * (Q_3 - Q_2) * (((T - T_b_C)**2 / (T_cr_C - T_b_C)**3) -
                                 ((T - T_b_C) / (T_cr_C - T_b_C)**2)))
    dadT_vals = conditional(lt(T, T_b_C), cond_1, cond_2)
    return dadT_vals

def da_carbdT(T_k):
    '''
    Derviative of the parameter a with respect to
    temperature, dadT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dadT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter a with respect to
    temperature.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_b_C = 100
    T_cr_C = T_cr - T_0_K
    cond_1 = 0
    cond_2 = (6 * (Q_3_carb - Q_2) * (((T - T_b_C)**2 / (T_cr_C - T_b_C)**3) -
                                 ((T - T_b_C) / (T_cr_C - T_b_C)**2)))
    da_carbdT_vals = conditional(lt(T, T_b_C), cond_1, cond_2)
    return da_carbdT_vals

def dEdT(T_k):
    '''
    Derviative of the parameter E with respect to
    temperature, dEdT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dEdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter E with respect to
    temperature.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    T_ref_1_C = T_ref_1 - T_0_K
    cond_1 = (N * ((T_cr_C - T_ref_1_C)**N) / (T_cr_C - T)**(N + 1))
    cond_2 = (N / z * E_0)
    dEdT_vals = conditional(lt(T, T_cr_C), cond_1, cond_2)
    return dEdT_vals


def dGdT(T, Pc):
    '''
    Derviative of the parameter G with respect to
    temperature, dGdT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dGdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter G with respect to
    capillary pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    dGdT_vals = (b / (b - 1) * (E(T) / a(T) * Pc)**(1 / (b - 1)) *
                 (dEdT(T) * a(T) - E(T) * dadT(T)) / a(T)**2 * Pc)
    return dGdT_vals

def dG_carbdT(T, Pc):
    '''
    Derviative of the parameter G with respect to
    temperature, dGdT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dGdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter G with respect to
    capillary pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    dG_carbdT_vals = (b_carb / (b_carb - 1) * (E(T) / a_carb(T) * Pc)**(1 / (b_carb - 1)) *
                 (dEdT(T) * a_carb(T) - E(T) * da_carbdT(T)) / a_carb(T)**2 * Pc)
    return dG_carbdT_vals

def dS_ldT(Pg, Pc, T, ncaco3):
    '''
    Derviative of saturation after full carbonation as a function of temperature and
    capillary pressure, dS_ldT [1/K]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to temperature.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    dS_ldT_vals = conditional(lt(T, T_cr),
                              ((-1 / b * (G(T, Pc) + 1)**(-1 *
                                                         (1 + b) / b) *
                               dGdT(T, Pc)) * (1-ncaco3 / ncaco3_max)) 
                              + ((-1 / b_carb * (G_carb(T, Pc) + 1)**(-1 *
                                                          (1 + b_carb) / b_carb) *
                                dG_carbdT(T, Pc)) * (ncaco3 / ncaco3_max)),
                              0)
    return dS_ldT_vals

def dGdPc(T, Pc):
    '''
    Derviative of the parameter G with respect to
    capillary pressure, dGdPc [1/Pa]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dGdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter G with respect to
    capillary pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    dGdPc_vals = (b / (b - 1) * (E(T) / a(T) * Pc)**(1 / (b - 1)) *
                  E(T) / a(T))
    return dGdPc_vals

def dG_carbdPc(T, Pc):
    '''
    Derviative of the parameter G with respect to
    capillary pressure, dGdPc [1/Pa]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dGdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of parameter G with respect to
    capillary pressure.

    Notes
    -----
    There are currently 4 options for laws. SLOPTION == 1, 2 and 3 are
    based on the 'laws.pdf' file. SLOPTION == 4 has a dependence of
    temperature on Q3.
    Such options should be defined within 'materials_input.py'.
    '''
    dG_carbdPc_vals = (b_carb / (b_carb - 1) * (E(T) / a_carb(T) * Pc)**(1 / (b_carb - 1)) *
                                E(T) / a_carb(T)) 
    return dG_carbdPc_vals
def dS_ldPc(Pg, Pc, T, ncaco3):
    '''
    Derviative of saturation as a function of temperature and
    capillary pressure, dS_ldPc [1/Pa]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to capillary pressure.

    Notes
    -----
    '''
    dS_ldPc_vals = conditional(lt(T, T_cr),
                               (- 1 / b * (G(T, Pc) + 1)**(- (1 + b) / b) *
                                dGdPc(T, Pc))* (1 - ncaco3 / ncaco3_max) 
                               + (- 1 / b_carb * (G_carb(T, Pc) + 1)**(- (1 + b_carb) / b_carb) *
                                 dG_carbdPc(T, Pc))* (ncaco3 / ncaco3_max),
                               0)
    return dS_ldPc_vals

def dS_ld_ncaco3(Pg, Pc, T, ncaco3):
    '''
    Derviative of saturation as a function of degree of carbonation, dS_ldncaco3 [1/Pa]
    SLOPTION == 2
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldncaco3_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to degree of carbonation.

    Notes
    -----
    '''
    dS_ld_ncaco3_vals = (1 / ncaco3_max) * ( - (G(T, Pc) + 1)**(-1 / b) 
                                            + (G_carb(T, Pc) + 1)**(-1 / b_carb))
    return dS_ld_ncaco3_vals

def D_co2(Pg, Pc, T, phi, ncaco3):
    '''
    Diffusivity of CO2 gas in cement, D_co2 [m^2/s]

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
    
    phi [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    ncaco3 [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    D_co2_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the diffusivity of CO2 in cement.

    Notes
    -----
    Other diffusivity laws may be added.
    '''
    E_a = 13000
    D_co2_0 = 1.6e-5 * exp(- E_a / R / T)
    a = 2.74
    b = 4.2
    D_co2_vals =  D_co2_0 * (phi**a) * ((1-S_l(Pg, Pc, T, ncaco3))**b) / R / T 
    return D_co2_vals 

def rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    The molar carbonation rate of portlandite, rch [mol/m^3/s]

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
    
    phi [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rch_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the molar carbonation rate of portlandite.

    Notes
    -----
    '''
    rch_vals =    phi * S_l(Pg, Pc, T, ncaco3) * k_sl(Pg, Pc, T, ncaco3) * K_h(T) * to_ch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3) * Pco2 
    return rch_vals

def k_sl(Pg, Pc, T, ncaco3):
    '''
    The solid-liquid mass transfer coefficient, rch [/s]

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
    
    ncaco3[mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    phi [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    k_sl_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid-liquid mass transfer coefficient.

    Notes
    -----
    '''
    alpha_sl = 0
    #alpha_sl = 625
    k_sl_vals = 1/(1+(alpha_sl*((1-S_l(Pg, Pc, T, ncaco3))**4)))
    return k_sl_vals

def rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    The molar carbonation rate of portlandite, rcsh [mol/m^3/s]

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
    
    phi [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rcsh_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the molar carbonation rate of CSH.

    Notes
    -----
    '''
    rcsh_vals =   phi * S_l(Pg, Pc, T, ncaco3) * k_sl(Pg, Pc, T, ncaco3) * K_h(T) * to_csh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3)  * Pco2 
    return rcsh_vals
