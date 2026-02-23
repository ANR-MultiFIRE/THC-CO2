import sys, os
sys.path.append('../../')
from thermohygro.core.th_model import *
from thermohygro.core.constant_constitutive_laws import *
from thermohygro.materials.materials_constitutive_laws import *
import pathlib


# ============================= Batch Analysis ===============================


CHECK_BACTH = len(sys.argv)
if CHECK_BACTH > 1:
    BATCH = True
    print('Running in batch mode.')

else:
    BATCH = False
    print('Running in single simulation mode.')


# ======================== Directory for Output Files ========================

dir_output = '/dp_results'
dir_backup = '/backup'


# =========================== Time Discretization =============================
t = 0                      # Initial time
minutes = 60
hours = 60 * minutes
t_total = 12 * hours                # Total time in seconds
dt = 1                    # Time step
DT_PROTOCOL = {'dt_1':  30, 'n_days_1': 1/24,
                'dt_2': 30, 'n_days_2': 1/24, 'tau_2': 500,
                'dt_3': 30, 'n_days_3': 3/24, 'tau_3': 500,
                'dt_4': 30, 'tau_4': 500}
#DT_PROTOCOL = None
freq_out = 10           # Frequency of Writing the Fields on the Output Files


# =========================== Space Discretization ============================
# Mesh
lx = 0.028                        # Total length of the 1D domain
nx = 500                          # Number of cells


# ============================ Initial Conditions =============================
RH_0 = 0.98                        # Initial relative humidity
T_0 = 25+273.15                          # Initial temperature
Pg_0 = 101325                         # Initial gas pressure (ambient pressure)
Pc_0 = Pc_Keq(RH_0, T_0)              # Initial capillary pressure
Pco2_0 = 0.0                          # Initial CO2 pressure
nch_0 = 5600 
ncsh_0 = 2400 
ncaco3_0 = 2400
boundwater_0 = 0.42
phi_0 = 0.4
# ============================ Boundary Conditions ============================
q_bar_l = 0            # Liquid Mass Flux
q_bar_v = 0            # Water Vapor Flux
q_bar_a = 0            # Dry Air Flux
q_bar_T = 0            # Heat Flux
q_bar_CO2 = 0          # CO2 Flux
h_g = 0
h_g_h = 0.008 
h_g_c = 0.18
h_T = 20
h_T_h = 300
h_T_c = 8.3
h_co2 = 0.0015 
epsilon = 0.85
T_inf = 77 + 273.15          # Temperature at the far field sorrounding gas


def T_rad(t):
    #if t <= 130 * 60:
      #  rate = (460 - 400) / 130
     #   T = t * rate / 60 + 400 + 273.15
    #else:
        #T = 460 + 273.15
    T = 74+273.15
    return T


RH_inf = 0.01  # Relative Humidity far field sorrounding gas
Pc_inf = Pc_Keq(RH_inf, T_inf)
Pco2_inf = 1 * 101325 


# Densities at the far field sorrounding gas
rho_v_inf = rho_v(Pg_0, Pc_inf, T_0)
rho_a_inf = rho_a(Pg_0, Pc_inf, T_0)
rho_co2_inf = rho_co2(T_inf, Pco2_inf)
# Dictionary specifying  Boundary Conditions to be used
# Pg_BC options:

# 1. 'Dirichlet'
# 2. 'Neumann_conv'
# 3. 'Adiabatic'
Pg_BC = {'left': 'Neumann_conv', 'right': ''}

# Pc_BC options:
# 1. 'Dirichlet'
# 2. 'Neumann_conv'
# 3. 'Adiabatic'
Pc_BC = {'left': 'Neumann_conv', 'right': ''}

# T_BC options:
# 1. 'Dirichlet_hot'
# 2. 'Dirichlet_inf'
# 3. 'Neumann_conv'
# 4. 'Neumann_rad'
# 5. 'Neumann_conv_rad'
T_BC = {'left': 'Neumann_conv', 'right': ''}

# CO2_BC options:
# 1. 'Dirichlet'
# 3. 'Neumann_conv'
Pco2_BC = {'left': 'Neumann_conv', 'right': ''}


# ========================== Printing Initial Values ==========================
S_l_0 = float(S_l(Pg_0, Pc_Keq(RH_0, T_0), T_0, ncaco3_0))
S_l_inf = float(S_l(Pg_0, Pc_Keq(RH_inf, T_inf), T_inf, ncaco3_0))
print(f'S_l_0 = {round(S_l_0, 2)}')
print(f'S_l_inf = {round(S_l_inf, 2)}')
print(float(Pv(Pg_0, Pc_0, T_0)))
print(f'\\phi_0 = {phi_0}')
m_l_sat = 1 * float(rho_l(T_0)) * float(phi_0)
m_s = float(rho_s(T_0)) * (1 - float(phi_0))
tot_weight = (m_l_sat + m_s)
m_l_0 = float(S_l(Pg_0, Pc_Keq(RH_0, T_0), T_0, ncaco3_0)) * \
    float(rho_l(T_0)) * float(phi_0)
print(f'Weight_Loss_0 = {round((m_l_sat - m_l_0) / tot_weight * 100, 3)}%')
print(f'rho_v_inf = {round(float(rho_v(Pg_0, Pc_Keq(RH_inf, T_0), T_0)), 4)}\
       \nrho_a_inf = {round(float(rho_a(Pg_0, Pc_Keq(RH_inf, T_0), T_0)), 4)}')


# ============================ Running Simulation =============================
TH_MODEL = th_model_core(t_total, dt, nx, lx,
                         Pg_0, Pc_0, T_0, RH_0, Pco2_0, nch_0, ncsh_0, ncaco3_0, boundwater_0, phi_0,
                         Pg_BC, Pc_BC, T_BC, Pco2_BC,
                         h_g, h_T, h_co2, RH_inf, Pc_inf, T_inf, Pco2_inf,
                         q_bar_a, q_bar_l, q_bar_v, q_bar_T, q_bar_CO2,
                         dir_output, dir_backup, __file__,
                         DT_PROTOCOL=DT_PROTOCOL, freq_out=freq_out,
                         T_inf_hot=T_rad, epsilon=epsilon,
                         h_g_h=h_g_h, h_g_c=h_g_c, h_T_h=h_T_h, h_T_c=h_T_c)

# TH_MODEL = th_model_core_pc_zero(t_total, dt, nx, lx,
#                          Pg_0, Pc_0, T_0, RH_0,
#                          Pg_BC, Pc_BC, T_BC,
#                          h_g, h_T, RH_inf, Pc_inf, T_inf,
#                          q_bar_a, q_bar_l, q_bar_v, q_bar_T,
#                          dir_output, dir_backup, __file__,
#                          DT_PROTOCOL=DT_PROTOCOL, freq_out=freq_out,
#                          T_inf_hot=T_heater,
#                          h_g_h=h_g_h, h_g_c=h_g_c)
TH_MODEL.run()

print('\nThat\'s all folks!')
