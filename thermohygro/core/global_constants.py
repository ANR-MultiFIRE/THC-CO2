# Global Constants
from fenics import *

T_0_K         = Constant(273.15)          # Conversion to Kelvin Scale
T_cr          = Constant(647.3)           # Critical Temperature of Water
T_ref_1       = Constant(293.15)          # Reference Temperature 1 (Ambient)
T_ref_2       = Constant(298.15)          # Reference Temperature 2 (Ambient)
Pg_ref        = Constant(101325)          # Reference Gas Pressure (Ambient)
Cp_l          = Constant(4181)            # Heat Capacity of Liquid Water
Cp_v          = Constant(1805)            # Heat Capacity of Water Vapour
Cp_a          = Constant(1005.7)          # Heat Capacity of Dry Air
sigma_SB      = Constant(5.67e-8)         # Stefan-Boltzmann Constant
R             = Constant(8.31441)         # Ideal Gas Constant
M_v           = Constant(0.01801528)      # Molar Mass of Water Vapour
M_a           = Constant(0.0289645)       # Molar Mass of Dry Air
M_co2         = Constant(0.0440095)       # Molar Mass of gas CO2
D_v_0         = Constant(2.58e-5)         # Initial Mass Diffusivity



lam           = Constant(5e-7)
rch_0         = Constant(5e-6)
D_caco3       = Constant(5e-13)

vch           = Constant(33e-6)
vcc           = Constant(36e-6)
vcsh           = Constant(39e-6)
delta_v_ch    = Constant(3e-6)


M_cc          = Constant(0.040078)
M_ch          = Constant(0.0740927)
rho_ch        = Constant(2230)
rho_cem       = Constant(2120)


alpha_csh     = Constant(400)
beta          = Constant(30e-6)


nch_0 = Constant(5600)
ncsh_0 = Constant(2400)
n_0 = Constant(3.4)

ncaco3_max = Constant(13000)

Q             = Constant(48096)


delta_H_R   = Constant(3000)