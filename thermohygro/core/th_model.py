from fenics import *
from thermohygro.core.global_constants import *
from thermohygro.core.CK_blocks import *
from thermohygro.core.constant_constitutive_laws import *
from thermohygro.materials.materials_constitutive_laws import *
from os import path, mkdir
import shutil
from numpy import array, savez
from datetime import datetime
import numpy as np
import pathlib
core_dir = pathlib.Path(__file__).parent.absolute()
thermo_dir_ = pathlib.Path(core_dir).parent.absolute()
thermo_dir = str(thermo_dir_)
TH_Model_dir = str(pathlib.Path(thermo_dir_).parent.absolute())

'''
The core of the model. There are 5 classes, the first is the
base class, all the others are derived from it. They are:
1) class th_model_core(object)
2) class th_model_core_pc_zero(th_model_core)
3) class th_model_core_2D(th_model_core)
4) class th_model_core_2D_cylinder(th_model_core)
5) class th_model_core_3D(th_model_core):

1) Base class for 1D models
2) Model ignoring the contribution of Pc for 1D models
3) Model for 2D general cases
4) Model for 2D_cylinder for NT tests (possibly duplicated of 3)
5) Model for 3D general cases
'''


class th_model_core(object):
    def __init__(self, t_total, dt, nx, lx,
                 Pg_0, Pc_0, T_0, RH_0, Pco2_0, nch_0, ncsh_0, ncaco3_0, boundwater_0, phi_0,
                 Pg_BC, Pc_BC, T_BC, Pco2_BC,
                 h_g, h_T, h_co2, RH_inf, Pc_inf, T_inf, Pco2_inf,
                 q_bar_a, q_bar_l, q_bar_v, q_bar_T, q_bar_co2,
                 dir_output, dir_backup, case_input_file,
                 T_inf_hot=0, T_inf_cold=0, h_g_h=0, h_g_c=0,
                 h_T_h=0, h_T_c=0, epsilon=0,
                 freq_out=100, DT_PROTOCOL=None):
        self.t_total = t_total
        self.dt      = Constant(dt)
        self.nx      = nx
        self.lx      = lx
        self.Pg_0    = Pg_0
        self.Pc_0    = Pc_0
        self.T_0     = T_0
        self.Pco2_0     = Pco2_0
        self.nch_0   = nch_0
        self.ncsh_0   = ncsh_0
        self.ncaco3_0   = ncaco3_0
        self.boundwater_0   = boundwater_0   
        self.phi_0   = phi_0
        self.Pg_BC   = Pg_BC
        self.Pc_BC   = Pc_BC
        self.T_BC    = T_BC
        self.Pco2_BC    = Pco2_BC
        self.h_g     = h_g
        self.h_g_h   = h_g_h
        self.h_g_c   = h_g_c
        if h_g_h == 0:
            self.h_g_h = self.h_g
        if h_g_c == 0:
            self.h_g_c = self.h_g
        self.h_T     = h_T
        self.h_T_h   = h_T_h
        self.h_T_c   = h_T_c
        if h_T_h == 0:
            self.h_T_h = self.h_T
        if h_T_c == 0:
            self.h_T_c = self.h_T 
        self.h_co2   = h_co2
        self.RH_0    = RH_0
        self.RH_inf  = RH_inf
        self.Pc_inf  = Pc_inf
        self.T_inf   = T_inf
        self.Pco2_inf   = Pco2_inf
        self.q_bar_a = q_bar_a
        self.q_bar_l = q_bar_l
        self.q_bar_v = q_bar_v
        self.q_bar_T = q_bar_T
        self.q_bar_co2 = q_bar_co2  
        self.dir_output  = dir_output
        self.dir_backup  = dir_backup
        self.case_input_file = case_input_file
        self.T_inf_hot   = T_inf_hot
        self.T_inf_cold  = T_inf_cold
        self.DT_PROTOCOL = DT_PROTOCOL
        self.freq_out    = freq_out
        self.epsilon = epsilon

    def generate_mesh(self):
        '''
        Mesh generation on an interval [0, lx]
        '''
        self.mesh = IntervalMesh(self.nx, 0.0, self.lx)

    def mark_boundaries(self):
        lx = self.lx
        '''
        Boundaries definition through a meshfunction
        that attributes an int to nodes
        '''
        class left(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]) < DOLFIN_EPS and on_boundary

        class right(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - lx) < DOLFIN_EPS and on_boundary
        self.boundaries = MeshFunction('size_t', self.mesh,
                                       self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)           # Set all the nodes to 0
        self.left = left()
        self.right = right()
        self.left.mark(self.boundaries, 1)   # Mark the geom. boundary nodes
        self.right.mark(self.boundaries, 2)  # left w. 1 and on the right w. 2

    def generate_function_spaces(self):
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 2)  # 2nd Degree Polynomial FE
        element = MixedElement([P2, P2, P2, P2, P2, P2, P2, P2, P2])              # Mixed Element for the primery variables
        self.V = FunctionSpace(self.mesh, element)        # Function Space defined over mesh w. FE

    def generate_functions(self):
        self.u = Function(self.V)                         # Function defined over V
        self.Pg, self.Pc, self.T, self.Pco2, self.nch, self.ncsh, self.ncaco3, self.boundwater, self.phi = split(self.u)          # Spliting u into the 3 primary variables

    def generate_IC_functions(self):
        self.Pg_n = interpolate(Constant(self.Pg_0), self.V.sub(0).collapse())
        self.Pc_n = interpolate(Constant(self.Pc_0), self.V.sub(1).collapse())
        self.T_n = interpolate(Constant(self.T_0), self.V.sub(2).collapse())
        self.Pco2_n = interpolate(Constant(self.Pco2_0), self.V.sub(3).collapse())
        self.nch_n = interpolate(Constant(self.nch_0), self.V.sub(4).collapse())
        self.ncsh_n = interpolate(Constant(self.ncsh_0), self.V.sub(5).collapse())
        self.ncaco3_n = interpolate(Constant(self.ncaco3_0), self.V.sub(6).collapse())
        self.boundwater_n = interpolate(Constant(self.boundwater_0), self.V.sub(7).collapse())
        self.phi_n = interpolate(Constant(self.phi_0), self.V.sub(8).collapse())


    def generate_BCS(self):
        dirichlet_bcs = []
        if self.Pg_BC['left'] == 'Dirichlet':
            print('Adding Pg Dirichlet Boundary Condition on the Left')
            bc_l_Pg = DirichletBC(self.V.sub(0), self.Pg_0, self.left)
            dirichlet_bcs += [bc_l_Pg]

        if self.Pg_BC['right'] == 'Dirichlet':
            print('Adding Pg Dirichlet Boundary Condition on the Right')
            bc_r_Pg = DirichletBC(self.V.sub(0), self.Pg_0, self.right)
            dirichlet_bcs += [bc_r_Pg]

        if self.Pc_BC['left'] == 'Dirichlet':
            print('Adding Pc Dirichlet Boundary Condition on the Left')
            # Expression for "gently" introducing the BC for Pc
            self.Pc_inf_Exp  = Expression('Pc_inf', degree=2, Pc_inf=self.Pc_inf, w=1)
            bc_l_Pc = DirichletBC(self.V.sub(1), self.Pc_inf_Exp, self.left)
            dirichlet_bcs += [bc_l_Pc]

        if self.Pc_BC['right'] == 'Dirichlet':
            print('Adding Pc Dirichlet Boundary Condition on the Right')
            # Expression for "gently" introducing the BC for Pc
            self.Pc_inf_Exp = Expression('Pc_inf', degree=2, Pc_inf=self.Pc_inf, w=1)
            bc_r_Pc = DirichletBC(self.V.sub(1), self.Pc_inf_Exp, self.right)
            dirichlet_bcs += [bc_r_Pc]

        if self.T_BC['left'] == 'Dirichlet_hot':
            print('Adding T_hot Dirichlet Boundary Condition on the Left')
            self.T_inf_hot_Exp  = Expression('T', degree=2, T=self.T_inf_hot(0))
            bc_l_T = DirichletBC(self.V.sub(2), self.T_inf_hot_Exp, self.left)
            dirichlet_bcs += [bc_l_T]

        if self.T_BC['left'] == 'Dirichlet_inf':
            print('Adding T_inf Dirichlet Boundary Condition on the Left')
            self.T_inf_cold_Exp  = Expression('T', degree=2, T=self.T_inf_cold(0))
            bc_l_T = DirichletBC(self.V.sub(2), self.T_inf_cold_Exp, self.left)
            dirichlet_bcs += [bc_l_T]

        if self.T_BC['right'] == 'Dirichlet_hot':
            print('Adding T_hot Dirichlet Boundary Condition on the Right')
            self.T_inf_hot_Exp  = Expression('T', degree=2, T=self.T_inf_hot(0))
            bc_r_T = DirichletBC(self.V.sub(2), self.T_inf_hot_Exp, self.right)
            dirichlet_bcs += [bc_r_T]

        if self.T_BC['right'] == 'Dirichlet_inf':
            print('Adding T_inf Dirichlet Boundary Condition on the Right')
            self.T_inf_cold_Exp  = Expression('T', degree=2, T=self.T_inf_cold(0))
            bc_r_T = DirichletBC(self.V.sub(2), self.T_inf_cold_Exp, self.right)
            dirichlet_bcs += [bc_r_T]
            
        if self.Pco2_BC['left'] == 'Dirichlet':
            print('Adding Pco2 Dirichlet Boundary Condition on the Left')
            # Expression for "gently" introducing the BC for Pc
            self.Pco2_inf_Exp  = Expression('Pco2_inf', degree=2, Pco2_inf=self.Pco2_inf, w=1)
            bc_l_Pco2 = DirichletBC(self.V.sub(3), self.Pco2_inf_Exp, self.left)
            dirichlet_bcs += [bc_l_Pco2]

        if self.Pco2_BC['right'] == 'Dirichlet':
            print('Adding Pco2 Dirichlet Boundary Condition on the Right')
            # Expression for "gently" introducing the BC for Pc
            self.Pco2_inf_Exp = Expression('Pco2_inf', degree=2, Pco2_inf=self.Pco2_inf, w=1)
            bc_r_Pco2 = DirichletBC(self.V.sub(3), self.Pco2_inf_Exp, self.right)
            dirichlet_bcs += [bc_r_Pco2]

        self.dirichlet_bcs = dirichlet_bcs



    def define_total_residual(self):
        # Substitute the ds measure with a new measure w. the marked boundaries
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        v_1, v_2, v_3, v_4, v_5, v_6, v_7 , v_8, v_9 = TestFunctions(self.V)             # Test functions of V
        Pg, Pc, T, Pco2, nch, ncsh, ncaco3, boundwater, phi = split(self.u)
        Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, boundwater_n, phi_n = self.Pg_n, self.Pc_n, self.T_n, self.Pco2_n, self.nch_n, self.ncsh_n, self.ncaco3_n, self.boundwater_n, self.phi_n
        dt = self.dt
        h_T = self.h_T
        h_g_h = self.h_g_h
        h_g_c = self.h_g_c
        h_T_h = self.h_T_h
        h_T_c = self.h_T_c
        h_co2 = self.h_co2
        q_bar_a = self.q_bar_a
        q_bar_l = self.q_bar_l
        q_bar_v = self.q_bar_v
        q_bar_T = self.q_bar_T
        q_bar_co2 = self.q_bar_co2
        rho_v_inf = rho_v(self.Pg_0, self.Pc_inf, self.T_0)
        rho_a_inf = rho_a(self.Pg_0, self.Pc_inf, self.T_0)
        rho_co2_inf = rho_co2(self.T_inf, self.Pco2_inf)
        Pco2_inf = self.Pco2_inf
        T_inf = self.T_inf
        if type(self.T_inf_hot) != int:
            self.T_inf_hot_Exp_rad = Expression('T_inf_hot', degree=2, T_inf_hot=self.T_inf_hot(0))
        T_inf_cold = self.T_inf_cold
        epsilon = self.epsilon
        
        # "Dry Air Equation"
        MBA = C_gg(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((Pg - Pg_n) / dt) * v_1 * dx
        MBA += C_gc(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((Pc - Pc_n) / dt) * v_1 * dx
        MBA += C_gt(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((T - T_n) / dt) * v_1 * dx
        MBA += C_gf(Pg_n, Pc_n, T_n, ncaco3_n) * ((phi - phi_n) / dt) * v_1 * dx
        MBA += C_gn(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((ncaco3 - ncaco3_n) / dt) * v_1 * dx
        MBA += inner(K_gg(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pg), grad(v_1)) * dx
        MBA += inner(K_gc(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pc), grad(v_1)) * dx
        MBA += inner(K_gt(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(T), grad(v_1)) * dx
        if self.Pg_BC['left'] == 'Neumann_conv':
            print('Adding Pg Convection on the Left')
            MBA += h_g_h * drho_adPg(Pg_n, Pc_n, T_n) * (Pg - Pg_n) * \
                v_1 * ds(1)
            MBA += h_g_h * drho_adPc(Pg_n, Pc_n, T_n) * (Pc - Pc_n) * \
                v_1 * ds(1)
            MBA += h_g_h * drho_adT(Pg_n, Pc_n, T_n) * (T - T_n) * \
                v_1 * ds(1)
            MBA += - (q_bar_a - h_g_h * (rho_a(Pg_n, Pc_n, T_n) - rho_a_inf)) * \
                v_1 * ds(1)
        if self.Pg_BC['right'] == 'Neumann_conv':
            print('Adding Pg Convection on the Right')
            MBA += h_g_c * drho_adPg(Pg_n, Pc_n, T_n) * (Pg - Pg_n) * \
                v_1 * ds(2)
            MBA += h_g_c * drho_adPc(Pg_n, Pc_n, T_n) * (Pc - Pc_n) * \
                v_1 * ds(2)
            MBA += h_g_c * drho_adT(Pg_n, Pc_n, T_n) * (T - T_n) * \
                v_1 * ds(2)
            MBA += - (q_bar_a - h_g_c * (rho_a(Pg_n, Pc_n, T_n) - rho_a_inf)) * \
                v_1 * ds(2)

        # "Water Equation"
        MBH = C_cc(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((Pc - Pc_n) / dt) * v_2 * dx
        MBH += C_cg(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((Pg - Pg_n) / dt) * v_2 * dx
        MBH += C_ct(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((T - T_n) / dt) * v_2 * dx
        MBH += C_cf(Pg_n, Pc_n, T_n, ncaco3_n) * ((phi - phi_n) / dt) * v_2 * dx
        MBH += C_cn(Pg_n, Pc_n, T_n, ncaco3_n) * ((ncaco3 - ncaco3_n) / dt) * v_2 * dx
        MBH += f_h(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n) * v_2 * dx
        MBH += inner(K_cc(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pc), grad(v_2)) * dx
        MBH += inner(K_cg(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pg), grad(v_2)) * dx
        MBH += inner(K_ct(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(T), grad(v_2)) * dx
        if self.Pc_BC['left'] == 'Neumann_conv':
            print('Adding Pc Convection on the Left')
            MBH += h_g_h * drho_vdPg(Pg_n, Pc_n, T_n) * \
                (Pg - Pg_n) * v_2 * ds(1)
            MBH += h_g_h * drho_vdPc(Pg_n, Pc_n, T_n) * \
                (Pc - Pc_n) * v_2 * ds(1)
            MBH += h_g_h * drho_vdT(Pg_n, Pc_n, T_n) * \
                (T - T_n) * v_2 * ds(1)
            MBH += - (q_bar_l + q_bar_v - h_g_h *
                      (rho_v(Pg_n, Pc_n, T_n) - rho_v_inf)) * v_2 * ds(1)
        if self.Pc_BC['right'] == 'Neumann_conv':
            print('Adding Pc Convection on the Rigt')
            MBH += h_g_c * drho_vdPg(Pg_n, Pc_n, T_n) * \
                (Pg - Pg_n) * v_2 * ds(2)
            MBH += h_g_c * drho_vdPc(Pg_n, Pc_n, T_n) * \
                (Pc - Pc_n) * v_2 * ds(2)
            MBH += h_g_c * drho_vdT(Pg_n, Pc_n, T_n) * \
                (T - T_n) * v_2 * ds(2)
            MBH += - (q_bar_l + q_bar_v - h_g_c *
                      (rho_v(Pg_n, Pc_n, T_n) - rho_v_inf)) * v_2 * ds(2)

        # Energy conservation equation
        ECE = C_tt(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((T - T_n) / dt) * v_3 * dx
        ECE += C_tc(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * ((Pc - Pc_n) / dt) * v_3 * dx
        # ECE += C_tg = 0
        ECE += inner(K_tt(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(T), grad(v_3)) * dx
        ECE += inner(K_tc(Pg_n, Pc_n, T_n,phi_n, ncaco3_n) * grad(Pc), grad(v_3)) * dx
        ECE += inner(K_tg(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pg), grad(v_3)) * dx
        ECE += - (q_bar_T) * v_3 * (ds(1) + ds(2))
        if 'conv' in self.T_BC['left']:
            print('Adding T Convection on the Left')
            ECE += - (- h_T_h * (T - T_inf)) * v_3 * ds(1)

        if 'conv' in self.T_BC['right']:
            print('Adding T Convection on the Right')
            ECE += - (- h_T_c * (T - T_inf)) * v_3 * ds(2)

        if 'rad' in self.T_BC['left']:
            print('Adding T Radiation on the Left')
            ECE += - (- epsilon * sigma_SB *
                      (T_n**3 * T - T_inf**4)) * v_3 * ds(1)

        if 'rad' in self.T_BC['right']:
            print('Adding T Radiation on the Right')
            ECE += - (- epsilon * sigma_SB *
                      (T_n**3 * T - T_inf**4)) * v_3 * ds(2)

        if 'heater' in self.T_BC['left']:
            print('Adding Heater on the Left')
            ECE += - (- epsilon * sigma_SB *
                      (T_n**3 * T - self.T_inf_hot_Exp_rad**4)) * v_3 * ds(1)
            print('Adding T Convection on the Left')
            ECE += - (- h_T_h * (T - self.T_inf_hot_Exp_rad)) * v_3 * ds(1)

        if 'cooler' in self.T_BC['right']:
            print('Adding T Radiation on the Right')
            ECE += - (- epsilon * sigma_SB *
                      (T_n**3 * T - T_inf_cold**4)) * v_3 * ds(2)

        # CO2 equation
        MBO = C_oo(Pg_n, Pc_n, T_n, Pco2_n, phi_n, ncaco3_n) * ((Pco2 - Pco2_n) / dt) * v_4 * dx
        MBO += C_oh(Pg_n, Pc_n, T_n, Pco2_n, phi_n, ncaco3_n) * ((Pc - Pc_n) / dt) * v_4 * dx
        MBO += C_ot(Pg_n, Pc_n, T_n, Pco2_n, phi_n, ncaco3_n) * ((T - T_n) / dt) * v_4 * dx
        MBO += C_of(Pg_n, Pc_n, T_n, Pco2_n, ncaco3_n) * ((phi - phi_n) / dt) * v_4 * dx
        MBO += C_on(Pg_n, Pc_n, T_n, Pco2_n, phi_n, ncaco3_n) * ((ncaco3 - ncaco3_n) / dt) * v_4 * dx
        MBO += f_o(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n) * v_4 * dx
        MBO += inner(K_oo(Pg_n, Pc_n, T_n, phi_n, ncaco3_n) * grad(Pco2), grad(v_4)) * dx
        MBO += inner(K_og(Pg_n, Pc_n, T_n, Pco2_n, phi_n, ncaco3_n) * grad(Pco2), grad(v_4)) * dx
        #if self.Pco2_BC['left'] == 'Neumann_conv':
         #   print('Adding CO2 Convection on the Left')
          #  MBO += - ( h_co2 * (Pco2 - Pco2_inf)) * \
           #     v_4 * ds(1)
        if self.Pco2_BC['left'] == 'Neumann_conv':
            print('Adding Pco2 Convection on the Left')
            MBO += h_g_h * drho_co2dPco2(T_n, Pco2_n) * (Pco2 - Pco2_n) * \
                v_4 * ds(1)
            MBO +=  h_g_h * (rho_co2(T_n, Pco2_n) - rho_co2_inf) * \
                v_4 * ds(1)
                
       # CH equation
        MBP = C_pp(nch_n) * ((nch - nch_n) / dt) * v_5 * dx
        MBP += f_pp(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n) * v_5 * dx
        
       # CSH equation
        MBE = C_ee(ncsh_n) * ((ncsh - ncsh_n) / dt) * v_6 * dx
        MBE += f_ee(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n) * v_6 * dx
         
       # CaCO3 equation
        MBC = C_ca(ncaco3_n) * ((ncaco3 - ncaco3_n) / dt) * v_7 * dx
        MBC += f_ca(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n)* v_7 * dx
        
       # bound water equation  
        MBW = C_ww(boundwater_n) * ((boundwater - boundwater_n) / dt) * v_8 * dx
        MBW += f_ww(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n)* v_8 * dx

        # phi equation
        MBF = C_ff(phi_n) * ((phi - phi_n) / dt) *  v_9 * dx
        MBF += f_f(Pg_n, Pc_n, T_n, Pco2_n, nch_n, ncsh_n, ncaco3_n, phi_n) * v_9 * dx


        Res = MBH + MBA + ECE + MBO + MBP + MBE + MBC + MBW + MBF
        self.Res = Res

    def create_variational_problem_and_solver(self):
        parameters["form_compiler"]["cpp_optimize"] = True
        ffc_options = {"quadrature_degree": 6, "optimize": True}
        if has_linear_algebra_backend("Epetra"):
            parameters["linear_algebra_backend"] = "Epetra"

        Jac = derivative(self.Res, self.u)
        self.problem = NonlinearVariationalProblem(self.Res, self.u,
                                              self.dirichlet_bcs, Jac,
                                              ffc_options)
        self.solver = NonlinearVariationalSolver(self.problem)
        prm = self.solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = 1E-5
        prm["newton_solver"]["relative_tolerance"] = 1E-15
        prm["newton_solver"]["maximum_iterations"] = 10
        prm['newton_solver']['error_on_nonconvergence'] = True

        # Set FEniCS log level
        set_log_level(50)

    def prepare_outputs(self):
        dir_output = TH_Model_dir + '/results' + self.dir_output
        dir_backup = self.dir_backup
        if not path.exists(dir_output):
            mkdir(dir_output)
        if not path.exists(dir_output + dir_backup):
            mkdir(dir_output + dir_backup)

        shutil.copy(__file__,
                    dir_output + dir_backup + '/model.py')
        shutil.copy(self.case_input_file,
                    dir_output + dir_backup + '/case_input.py')
        shutil.copy(thermo_dir +'/materials/materials_constitutive_laws.py',
                    dir_output + dir_backup + '/materials_constitutive_laws.py')

        # Files for saving the fields
        self.filex = XDMFFile(dir_output + '/Fields.xdmf')
        self.filex.parameters['functions_share_mesh'] = True
        self.filex.parameters['rewrite_function_mesh'] = False
        self.filex.parameters["flush_output"] = True

    def simulation(self):
        t = 0            # Initial Time
        nt = 0           # Number of Time Steps
        Pgs = []         # Lists to Save the Nodal Values of Gas Pressure
        Pcs = []         # Lists to Save the Nodal Values of Capillary Pressure
        Ts = []          # Lists to Save the Nodal Values of Temperature
        Pco2s = []       # Lists to Save the Nodal Values of CO2
        nchs = []        # Lists to Save the Nodal Values of DoC
        ncshs  = []
        ncaco3s  = []
        boundwaters  = []
        phis = []        # Lists to Save the Nodal Values of phi
        ts = []          # Lists to Save the Time Step Size (for variable dt)
        if self.DT_PROTOCOL:
            dt_1 = self.DT_PROTOCOL['dt_1']
            ndays_1 = self.DT_PROTOCOL['n_days_1']
            dt_2 = self.DT_PROTOCOL['dt_2']
            ndays_2 = ndays_1 + self.DT_PROTOCOL['n_days_2']
            tau_2 = self.DT_PROTOCOL['tau_2']
            dt_3 = self.DT_PROTOCOL['dt_3']
            ndays_3 = ndays_2 + self.DT_PROTOCOL['n_days_3']
            tau_3 = self.DT_PROTOCOL['tau_3']
            dt_4 = self.DT_PROTOCOL['dt_4']
            tau_4 = self.DT_PROTOCOL['tau_4']

        # Calculate time of simulation
        startTime = datetime.now()
        # Loop over timesteps
        while t <= self.t_total:
            # Solve non-linear problem
            if self.DT_PROTOCOL:
                if t < ndays_1 * 24 * 3600:
                    self.dt.assign(dt_1)
                elif t < ndays_2 * 24 * 3600:
                    tau = tau_2
                    if round(float(self.dt)) < dt_2:
                        self.dt.assign(float(self.dt) *
                                       np.exp(np.log(dt_2 / dt_1) / tau))
                elif t < ndays_3 * 24 * 3600:
                    tau = tau_3
                    if round(float(self.dt)) < dt_3:
                        self.dt.assign(float(self.dt) *
                                       np.exp(np.log(dt_3 / dt_2) / tau))
                else:
                    tau = tau_4
                    if round(float(self.dt)) < dt_4:
                        self.dt.assign(float(self.dt) *
                                       np.exp(np.log(dt_4 / dt_3) / tau))
            if 'heater' in self.T_BC['left']:
                self.T_inf_hot_Exp_rad.T_inf_hot = self.T_inf_hot(t)

            if 'Dirichlet_hot' in self.T_BC['left']:
                self.T_inf_hot_Exp.T = self.T_inf_hot(t)

            if 'Dirichlet_inf' in self.T_BC['right']:
                self.T_inf_cold_Exp.T = self.T_inf_cold(t)

            if (('Dirichlet' in self.Pc_BC['left']) or
                ('Dirichlet' in self.Pc_BC['right'])):
                if t < 0.05 * 3600:
                    self.Pc_inf_Exp.w = 2
                elif t < 0.15 * 3600:
                    self.Pc_inf_Exp.w = 3
                elif t < 0.25 * 3600:
                    self.Pc_inf_Exp.w = 4
                elif t < 0.35 * 3600:
                    self.Pc_inf_Exp.w = 8
                elif t < 0.5 * 3600:
                    self.Pc_inf_Exp.w = 9.35

            # Number of iterations and bool variable to check if it converged
            # n, conv = self.solver.solve()
            try:
                n, conv = self.solver.solve()
            except:
                print("\nDidn\'t converge!")
                t = self.t_total
                
            # Sketch of an automatic substepping algorithm - seems to work
            # j = 0
            # conv = False
            # while (conv is False) or (j>20):
            #     try:
            #         n, conv = self.solver.solve()
            #         #print(conv)
            #     except:
            #         print("Not converging with current dt")
            #         dt_1 = float(self.dt/2)
            #         self.dt.assign(dt_1)
            #         self.generate_functions()
            #         self.generate_IC_functions()
            #         self.define_total_residual()
            #         self.create_variational_problem_and_solver()
            #         print(f'New dt: {float(self.dt)}')
            #         j += 1
            #         pass

            # Update solution with last computed value
            (_Pg, _Pc, _T, _Pco2, _nch, _ncsh, _ncaco3, _boundwater, _phi) = self.u.split(True)
            self.Pg_n.vector()[:] = _Pg.vector()
            self.Pc_n.vector()[:] = _Pc.vector()
            self.T_n.vector()[:] = _T.vector()
            self.Pco2_n.vector()[:] = _Pco2.vector()
            self.nch_n.vector()[:] = _nch.vector()
            self.ncsh_n.vector()[:] = _ncsh.vector()
            self.ncaco3_n.vector()[:] = _ncaco3.vector()
            self.boundwater_n.vector()[:] = _boundwater.vector()
            self.phi_n.vector()[:] = _phi.vector()


            # Create lists of the nodal values
            Pgs.append(_Pg.compute_vertex_values())
            Pcs.append(_Pc.compute_vertex_values())
            Ts.append(_T.compute_vertex_values())
            Pco2s.append(_Pco2.compute_vertex_values())
            nchs.append(_nch.compute_vertex_values())
            ncshs.append(_ncsh.compute_vertex_values())
            ncaco3s.append(_ncaco3.compute_vertex_values())
            boundwaters.append(_boundwater.compute_vertex_values())
            phis.append(_phi.compute_vertex_values())

            # List of time steps for variable dt values
            ts.append(float(self.dt))

            # Calculation of min and max values of primary variables
            Pg_max = round(max(self.Pg_n.vector()[:]), 3)
            Pg_min = round(min(self.Pg_n.vector()[:]), 3)

            Pc_max = round(max(self.Pc_n.vector()[:]) / 1e6, 3)
            Pc_min = round(min(self.Pc_n.vector()[:]) / 1e6, 3)

            T_max = round(max(self.T_n.vector()[:]), 3)
            T_min = round(min(self.T_n.vector()[:]), 3)
            
            Pco2_max = round(max(self.Pco2_n.vector()[:]), 3)
            Pco2_min = round(min(self.Pco2_n.vector()[:]), 3)
            
            nch_max = round(max(self.nch_n.vector()[:]), 3)
            nch_min = round(min(self.nch_n.vector()[:]), 3)

            ncsh_max = round(max(self.ncsh_n.vector()[:]), 3)
            ncsh_min = round(min(self.ncsh_n.vector()[:]), 3)

            ncaco3_max = round(max(self.ncaco3_n.vector()[:]), 3)
            ncaco3_min = round(min(self.ncaco3_n.vector()[:]), 3)
            
            boundwater_max = round(max(self.boundwater_n.vector()[:]), 3)
            boundwater_min = round(min(self.boundwater_n.vector()[:]), 3)

            phi_max = round(max(self.phi_n.vector()[:]), 3)
            phi_min = round(min(self.phi_n.vector()[:]), 3)

            # Printing informations
            if self.DT_PROTOCOL:
                print(f'''
                      +---------------------------------------+
                      | Progress   :      {str(t / self.t_total * 100)[:7]:>7} %           |
                      |---------------------------------------|
                      | T_min      :    {str(T_min)[:10]:>10} K          |
                      | T_max      :    {str(T_max)[:10]:>10} K          |
                      | Pg_min     :    {str(Pg_min)[:10]:>10} Pa         |
                      | Pg_max     :    {str(Pg_max)[:10]:>10} Pa         |
                      | Pco2_min   :    {str(Pco2_min)[:10]:>10} Pa         |
                      | Pco2_max   :    {str(Pco2_max)[:10]:>10} Pa         |
                      | Pc_min     :    {str(Pc_min)[:10]:>10} MPa        |
                      | Pc_max     :    {str(Pc_max)[:10]:>10} MPa        |
                      | nch_min    :    {str(nch_min)[:10]:>10} mol/m3     |  
                      | nch_max    :    {str(nch_max)[:10]:>10} mol/m3     |
                      | w_min      :    {str(boundwater_min)[:10]:>10}            |  
                      | w_max      :    {str(boundwater_max)[:10]:>10}            |
                      | ncsh_min   :    {str(ncsh_min)[:10]:>10} mol/m3     |
                      | ncsh_max   :    {str(ncsh_max)[:10]:>10} mol/m3     |
                      | ncaco3_min :    {str(ncaco3_min)[:10]:>10} mol/m3     |
                      | ncaco3_max :    {str(ncaco3_max)[:10]:>10} mol/m3     |
                      | phi_min    :    {str(phi_min)[:10]:>10}            |
                      | phi_max    :    {str(phi_max)[:10]:>10}            |
                      | t[h]       :    {str(float(t/3600))[:6]:>6}                |
                      | dt         :    {float(self.dt):6.2f}                |
                      +---------------------------------------+ 
                      ''', end='\r')
            else:
                print(f'''
                      +---------------------------------------+
                      | Progress   :      {str(t / self.t_total * 100)[:7]:>7} %           |
                      |---------------------------------------|
                      | T_min      :    {str(T_min)[:10]:>10} K          |
                      | T_max      :    {str(T_max)[:10]:>10} K          |
                      | Pg_min     :    {str(Pg_min)[:10]:>10} Pa         |
                      | Pg_max     :    {str(Pg_max)[:10]:>10} Pa         |
                      | Pco2_min   :    {str(Pco2_min)[:10]:>10} Pa         |
                      | Pco2_max   :    {str(Pco2_max)[:10]:>10} Pa         |
                      | Pc_min     :    {str(Pc_min)[:10]:>10} MPa        |
                      | Pc_max     :    {str(Pc_max)[:10]:>10} MPa        |
                      | nch_min    :    {str(nch_min)[:10]:>10} mol/m3     |  
                      | nch_max    :    {str(nch_max)[:10]:>10} mol/m3     |
                      | w_min      :    {str(boundwater_min)[:10]:>10}            |  
                      | w_max      :    {str(boundwater_max)[:10]:>10}            |
                      | ncsh_min   :    {str(ncsh_min)[:10]:>10} mol/m3     |
                      | ncsh_max   :    {str(ncsh_max)[:10]:>10} mol/m3     |
                      | ncaco3_min :    {str(ncaco3_min)[:10]:>10} mol/m3     |
                      | ncaco3_max :    {str(ncaco3_max)[:10]:>10} mol/m3     |
                      | phi_min    :    {str(phi_min)[:10]:>10}            |
                      | phi_max    :    {str(phi_max)[:10]:>10}            |
                      | t[h]       :    {str(float(t/3600))[:6]:>6}                |
                      +---------------------------------------+ 
                      ''', end='\r')



            # Saving the files at the frequency of output into the output files
            if (nt % self.freq_out == 0):
                _Pg.rename("Gas Pressure [Pa]", "Pg")
                _Pc.rename("Capillary Pressure [Pa]", "Pc")
                _T.rename("Temperature [K]", "T")
                _Pco2.rename("CO2 Pressure [Pa]", "Pco2")
                _nch.rename("nch [-]", "nch")
                _ncsh.rename("ncsh [-]", "ncsh")
                _ncaco3.rename("ncaco3 [-]", "ncaco3")
                _boundwater.rename("boundwater [-]", "w")
                _phi.rename("Porosity [-]", "phi")

                self.filex.write(_Pg, t)
                self.filex.write(_Pc, t)
                self.filex.write(_T, t)
                self.filex.write(_Pco2, t)
                self.filex.write(_nch, t)
                self.filex.write(_ncsh, t)
                self.filex.write(_ncaco3, t)
                self.filex.write(_boundwater, t)
                self.filex.write(_phi, t)


            nt += 1
            t += float(self.dt)
        # End loop over time steps
        # Calculation of the CPU time
        time_delta = datetime.now() - startTime
        print('\nSimulation time: ', str(time_delta))
        self.Pgs = Pgs
        self.Pcs = Pcs
        self.Ts = Ts
        self.Pco2s = Pco2s
        self.ts = ts
        self.nchs = nchs
        self.ncshs = ncshs
        self.ncaco3s = ncaco3s
        self.boundwaters = boundwaters
        self.phis = phis

    def write_output(self):
        # Converting into numpy arrays to save the files as .npz files
        Pgs = array(self.Pgs)
        Pcs = array(self.Pcs)
        Ts = array(self.Ts)
        Pco2s = array(self.Pco2s)
        nchs = array(self.nchs)
        ncshs = array(self.ncshs)
        ncaco3s = array(self.ncaco3s)
        boundwaters = array(self.boundwaters)
        phis = array(self.phis)
        ts = array(self.ts)
        dt = float(self.dt)
        # Creating the .npz files
        dir_output = TH_Model_dir + '/results' + self.dir_output
        savez(path.join(dir_output, 'Pgs_dt_' + str(dt)), Pgs=Pgs)
        savez(path.join(dir_output, 'Pcs_dt_' + str(dt)), Pcs=Pcs)
        savez(path.join(dir_output, 'Ts_dt_' + str(dt)), Ts=Ts)
        savez(path.join(dir_output, 'Pco2s_dt_' + str(dt)), Pco2s=Pco2s)
        savez(path.join(dir_output, 'time_s_dt_' + str(dt)), ts=ts)
        savez(path.join(dir_output, 'nchs_dt_' + str(dt)), nchs=nchs)
        savez(path.join(dir_output, 'ncshs_dt_' + str(dt)), ncshs=ncshs)
        savez(path.join(dir_output, 'ncaco3s_dt_' + str(dt)), ncaco3s=ncaco3s)
        savez(path.join(dir_output, 'boundwaters_dt_' + str(dt)), boundwaters=boundwaters)
        savez(path.join(dir_output, 'phis_dt_' + str(dt)), phis=phis)


    def run(self):
        self.generate_mesh()
        self.mark_boundaries()
        self.generate_function_spaces()
        self.generate_functions()
        self.generate_IC_functions()
        self.generate_BCS()
        self.define_total_residual()
        self.create_variational_problem_and_solver()
        self.prepare_outputs()
        self.simulation()
        self.write_output()
