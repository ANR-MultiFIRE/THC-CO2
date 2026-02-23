from fenics import *
from thermohygro.core.constant_constitutive_laws import *
from thermohygro.core.global_constants import *
from thermohygro.materials.materials_constitutive_laws import *


# Coefficients
def C_cc(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to P_c.
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
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    ncaco3 [mol/m3] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.


    Returns
    -------
    C_cc_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_cc_vals = (phi * ((1 - S_l(Pg, Pc, T, ncaco3)) * drho_vdPc(Pg, Pc, T)
                      + (rho_l(T) - rho_v(Pg, Pc, T)) * dS_ldPc(Pg, Pc, T, ncaco3)))
    return C_cc_vals


def C_cg(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to P_g.
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
    C_cg_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model at 
    moderate temperature for cementitious materials
    '''
    C_cg_vals = ((1 - S_l(Pg, Pc, T, ncaco3)) * phi * drho_vdPg(Pg, Pc, T))
    return C_cg_vals


def C_ct(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to T.
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
    C_ct_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ct_vals = (- dm_dehyddT(T)
                 + phi * ((rho_l(T) - rho_v(Pg, Pc, T)) * dS_ldT(Pg, Pc, T, ncaco3)
                 + (1 - S_l(Pg, Pc, T, ncaco3)) * drho_vdT(Pg, Pc, T)
                 + S_l(Pg, Pc, T, ncaco3) * drho_ldT(T)))
    return C_ct_vals

def C_cf(Pg, Pc, T, ncaco3):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to phi.
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
        
    ncaco3 [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    C_cf_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    phi.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_cf_vals =  (rho_l(T) * S_l(Pg, Pc, T, ncaco3) + rho_v(Pg, Pc, T)
                              * (1 - S_l(Pg, Pc, T, ncaco3)))
    return C_cf_vals

def C_cn(Pg, Pc, T, ncaco3):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to ncaco3.
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
        
    ncaco3 [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    C_cn_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    ncaco3.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_cn_vals =  (rho_l(T) - rho_v(Pg, Pc, T)) * dS_ld_ncaco3(Pg, Pc, T, ncaco3)
    return C_cn_vals

def f_h(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of P_c
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
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
    f_h_vals :  array_like, UFL function
    Source term of P_c

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_h_vals =  - M_v  * rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)
    return f_h_vals

def K_cc(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to P_c.
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
    K_cc_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_cc_vals = - (K(T,phi, ncaco3) * (rho_l(T) * k_rl(Pg, Pc, T, ncaco3) / mu_l(T))
                   + D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T) *
                   rho_v(Pg, Pc, T) / rho_l(T))
    return K_cc_vals


def K_cg(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to P_g.
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
    K_cg_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_cg_vals = (K(T,phi, ncaco3) * (rho_l(T) * k_rl(Pg, Pc, T, ncaco3) / mu_l(T)
                             + rho_v(Pg, Pc, T) * k_rg(Pg, Pc, T, ncaco3)
                             / mu_g(Pg, Pc, T))
                 + D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T)
                 * (rho_v(Pg, Pc, T) / rho_l(T) - Pv(Pg, Pc, T) / Pg))
    return K_cg_vals


def K_ct(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to T.
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
    K_ct_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_ct_vals = (D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T)
                 * dPvdT(Pg, Pc, T))
    return K_ct_vals


def C_gg(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to P_g.
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
    C_gg_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_gg_vals = ((1 - S_l(Pg, Pc, T, ncaco3)) * phi * drho_adPg(Pg, Pc, T))
    return C_gg_vals


def C_gc(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to P_c.
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
    C_gc_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_gc_vals = ((1 - S_l(Pg, Pc, T, ncaco3)) * phi * drho_adPc(Pg, Pc, T)
                 - rho_a(Pg, Pc, T) * phi * dS_ldPc(Pg, Pc, T, ncaco3))
    return C_gc_vals


def C_gt(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to T.
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
    C_gt_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_gt_vals = (phi * ((1 - S_l(Pg, Pc, T, ncaco3)) * drho_adT(Pg, Pc, T)
                             - rho_a(Pg, Pc, T) * dS_ldT(Pg, Pc, T, ncaco3)))
    return C_gt_vals

def C_gf(Pg, Pc, T, ncaco3):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to phi.
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
    C_gf_vals :  array_like, UFL function
    The capacitance matrix of P_g with terms differentiated with respect to 
    phi.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_gf_vals = (rho_a(Pg, Pc, T) * (1 - S_l(Pg, Pc, T, ncaco3)))
    return C_gf_vals

def C_gn(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to ncaco3.
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
    C_gn_vals :  array_like, UFL function
    The capacitance matrix of P_g with terms differentiated with respect to 
    ncaco3.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_gn_vals = - phi * rho_a(Pg, Pc, T) * dS_ld_ncaco3(Pg, Pc, T, ncaco3)
    return C_gn_vals

def K_gg(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to T.
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
    K_gg_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_gg_vals = (K(T,phi, ncaco3) * (rho_a(Pg, Pc, T) * k_rg(Pg, Pc, T, ncaco3) 
                             / mu_g(Pg, Pc, T))
                 - D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T) *
                 (rho_v(Pg, Pc, T) / rho_l(T) - Pv(Pg, Pc, T) / Pg))
    return K_gg_vals


def K_gc(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to P_c.
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
    K_gc_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    P_c.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_gc_vals = (D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T) *
                 (rho_v(Pg, Pc, T) / rho_l(T)))
    return K_gc_vals


def K_gt(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to T.
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
    K_gt_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_gt_vals = - (D_eff(Pg, Pc, T, phi, ncaco3) * (M_v * M_a * M_co2) / (M_g(Pg, Pc, T)**2 * R * T) *
                   dPvdT(Pg, Pc, T))
    return K_gt_vals


def C_tt(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of T with terms differentiated with respect to T.
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
    C_tt_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_tt_vals = (- H_vap(T) * (dm_dehyddT(T) 
                               + phi * (S_l(Pg, Pc, T, ncaco3) * drho_ldT(T) 
                                           + rho_l(T) * dS_ldT(Pg, Pc, T, ncaco3)))
                 + dm_dehyddT(T) * H_dehyd
                 + rhoCp(Pg, Pc, T, phi, ncaco3))
    return C_tt_vals


def C_tc(Pg, Pc, T, phi, ncaco3):
    '''
    Capacitance Matrix of T with terms differentiated with respect to P_c.
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
    C_tc_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    P_c.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_tc_vals = - (H_vap(T) * rho_l(T) * phi * dS_ldPc(Pg, Pc, T, ncaco3))
    return C_tc_vals


def C_tg(Pg, Pc, T):
    '''
    Capacitance Matrix of T with terms differentiated with respect to P_g.
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
    C_tg_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    P_g.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_tg_vals = 0
    return C_tg_vals


def K_tt(Pg, Pc, T, phi, ncaco3):
    '''
    Conductance Matrix of T with terms differentiated with respect to T.
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
    K_ct_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    return lambda_eff(Pg, Pc, T, phi, ncaco3)

def K_tc(Pg, Pc, T,phi, ncaco3):
    '''
    Conductance Matrix of T with terms differentiated with respect to P_c.
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
    K_tc_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    Pc.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_tc_vals = (H_vap(T) * K(T,phi, ncaco3) * rho_l(T) * k_rl(Pg, Pc, T, ncaco3) / mu_l(T))
    return K_tc_vals


def K_tg(Pg, Pc, T,phi, ncaco3):
    '''
    Conductance Matrix of T with terms differentiated with respect to g.
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
    K_tg_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    P_g.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    K_tg_vals = - (H_vap(T) * K(T,phi, ncaco3) * rho_l(T) * k_rl(Pg, Pc, T, ncaco3) / mu_l(T))
    return K_tg_vals


def C_oo(Pg, Pc, T, Pco2, phi, ncaco3):
    '''
    Capacitance Matrix of Pco2 with terms differentiated with respect to Pco2.
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
    C_oo_vals :  array_like, UFL function
    The capacitance matrix of Pco2 with terms differentiated with respect to 
    Pco2.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_oo_vals = ((phi) * ((1-S_l(Pg, Pc, T, ncaco3)))) * drho_co2dPco2(T, Pco2)
    return C_oo_vals

def C_ot(Pg, Pc, T, Pco2, phi, ncaco3):
    '''
    Capacitance Matrix of Pco2 with terms differentiated with respect to T.
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
    C_ot_vals :  array_like, UFL function
    The capacitance matrix of Pco2 with terms differentiated with respect to 
    T.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ot_vals = ((1-S_l(Pg, Pc, T, ncaco3)) * phi * drho_co2dT(T, Pco2) - rho_co2(T, Pco2) * phi * dS_ldT(Pg, Pc, T, ncaco3))
    return C_ot_vals

def C_oh(Pg, Pc, T, Pco2, phi, ncaco3):
    '''
    Capacitance Matrix of Pco2 with terms differentiated with respect to Pc.
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
    C_oh_vals :  array_like, UFL function
    The capacitance matrix of Pco2 with terms differentiated with respect to 
    Pc.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_oh_vals =  - phi * rho_co2(T, Pco2)  * dS_ldPc(Pg, Pc, T, ncaco3)
    return C_oh_vals

def C_of(Pg, Pc, T, Pco2, ncaco3):
    '''
    Capacitance Matrix of Pco2 with terms differentiated with respect to phi.
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
    C_of_vals :  array_like, UFL function
    The capacitance matrix of Pco2 with terms differentiated with respect to 
    phi.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_of_vals = ((1-S_l(Pg, Pc, T, ncaco3))) * rho_co2(T, Pco2)
    return C_of_vals

def C_on(Pg, Pc, T, Pco2, phi, ncaco3):
    '''
    Capacitance Matrix of Pco2 with terms differentiated with respect to ncaco3.
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
    C_on_vals :  array_like, UFL function
    The capacitance matrix of Pco2 with terms differentiated with respect to 
    ncaco3.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_on_vals = -  phi * rho_co2(T, Pco2)  * dS_ld_ncaco3(Pg, Pc, T, ncaco3)
    return C_on_vals

def f_o(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of Pco2
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
    f_o_vals :  array_like, UFL function
    Source term of Pco2

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_o_vals =  (rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)  +  n_0 * rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)) * M_co2 / R / T
    return f_o_vals

def K_oo(Pg, Pc, T, phi, ncaco3):
    K_oo_vals = D_co2(Pg, Pc, T, phi, ncaco3) * rho_g(Pg, Pc, T) * M_v * M_a * M_co2 / M_g(Pg, Pc, T)**3 / R / T
    return K_oo_vals

def K_og(Pg, Pc, T, Pco2, phi, ncaco3):
    K_og_vals = D_co2(Pg, Pc, T, phi, ncaco3) * rho_g(Pg, Pc, T) * M_v * M_a * M_co2 / M_g(Pg, Pc, T)**3 * Pco2 / Pg / R / T
    return K_og_vals


def C_pp(nch):
    '''
    Capacitance Matrix of nch with terms differentiated with respect to nch.
    Parameters
    ----------

    nch [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    C_pp_vals :  array_like, UFL function
    The capacitance matrix of nch with terms differentiated with respect to 
    nch.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_pp_vals = 1
    return C_pp_vals

def f_pp(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of nch
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
    f_pp_vals :  array_like, UFL function
    Source term of Pco2

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_pp_vals =  rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)
    return f_pp_vals

def C_ee(ncsh):
    '''
    Capacitance Matrix of ncsh with terms differentiated with respect to ncsh.
    Parameters
    ----------

    ncsh [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    C_ee_vals :  array_like, UFL function
    The capacitance matrix of ncsh with terms differentiated with respect to 
    ncsh.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ee_vals = 1
    return C_ee_vals

def f_ee(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of ncsh
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
    f_ee_vals :  array_like, UFL function
    Source term of ncsh

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_ee_vals =  rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)
    return f_ee_vals

def C_ca(ncaco3):
    '''
    Capacitance Matrix of ncaco3 with terms differentiated with respect to ncaco3.
    Parameters
    ----------

    ncaco3 [mol/m3] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    C_ca_vals :  array_like, UFL function
    The capacitance matrix of ncaco3 with terms differentiated with respect to 
    ncaco3.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ca_vals = 1
    return C_ca_vals

def f_ca(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of ncaco3
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
    f_ca_vals :  array_like, UFL function
    Source term of ncaco3

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_ca_vals = - (rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi) + n_0 * rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi))
    return f_ca_vals


def C_ww(w):
    '''
    Capacitance Matrix of w with terms differentiated with respect to w.
    Parameters
    ----------

    w [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    C_ww_vals :  array_like, UFL function
    The capacitance matrix of nch with terms differentiated with respect to 
    nch.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ww_vals = 1
    return C_ww_vals

def f_ww(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of w
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
    f_ww_vals :  array_like, UFL function
    Source term of w

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_ww_vals =  (M_v * (rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi) + n_0*rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi))) / rho_cem 
    return f_ww_vals

def C_ff(phi):
    '''
    Capacitance Matrix of phi with terms differentiated with respect to phi.
    Parameters
    ----------

    phi [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.
    
    Returns
    -------
    C_ff_vals :  array_like, UFL function
    The capacitance matrix of nch with terms differentiated with respect to 
    nch.

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    C_ff_vals = 1
    return C_ff_vals

def f_f(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi):
    '''
    Source term of phi
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
    f_f_vals :  array_like, UFL function
    Source term of phi

    Notes
    -----
    Equation in El Faqir et al, Coupled drying-carbonation model  
    at moderate temperature for cementitious materials
    '''
    f_f_vals = (delta_v_ch * rch(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)  + n_0*vcc*rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi)
    - beta * rcsh(Pg, Pc, T, Pco2, nch, ncsh, ncaco3, phi))
    return f_f_vals