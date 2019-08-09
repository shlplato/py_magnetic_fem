import os
import numpy as np
import femm
import time


class EcoreGeometry:
    def __init__(self, A, B, C, D, E, F):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F

    def get_XY_coord(self):
        # Outputs vectors [x,y] of the coordinate points that define the cores as polygons
        # [x_E] = E core
        # [x_I] = I core

        # E-core
        xy_vec = [[0, 0],
                  [0, self.D],
                  [(self.A - self.B) / 2, self.D],
                  [(self.A - self.B) / 2, (self.D - self.E)],
                  [(self.A - self.C) / 2, (self.D - self.E)],
                  [(self.A - self.C) / 2, self.D],
                  [(self.A + self.C) / 2, self.D],
                  [(self.A + self.C) / 2, (self.D - self.E)],
                  [(self.A + self.C) / 2, (self.D - self.E)],
                  [(self.A + self.B) / 2, (self.D - self.E)],
                  [(self.A + self.B) / 2, (self.D)],
                  [self.A, self.D],
                  [self.A, 0]]
        x_vec = xy_vec[:, 0]
        y_vec = xy_vec[:, 1]

        return [x_vec, y_vec]

    def draw_XY_geometry(self, headerFEM):
        headerFEM.create_polygon(np.array([[0, 0],
                                           [0, self.D],
                                           [(self.A - self.B) / 2, self.D],
                                           [(self.A - self.B) / 2, self.D - self.E],
                                           [(self.A - self.C) / 2, self.D - self.E],
                                           [(self.A - self.C) / 2, self.D],
                                           [(self.A + self.C) / 2, self.D],
                                           [(self.A + self.C) / 2, self.D - self.E],
                                           [(self.A + self.C) / 2, self.D - self.E],
                                           [(self.A + self.B) / 2, self.D - self.E],
                                           [(self.A + self.B) / 2, self.D],
                                           [self.A, self.D], [self.A, 0]]))


class Icore:
    def __init__(self, A, F, t):
        self.A = A
        self.F = F
        self.t = t

    def get_XY_coord(self):
        # Icore
        xy_vec = [[0, (self.D + self.g)],
                  [0, (self.D + self.g + self.It)],
                  [(self.A), (self.D + self.g + self.It)],
                  [(self.A), (self.D + self.g)]]

        x_vec = xy_vec[:, 0]
        y_vec = xy_vec[:, 1]

    def draw_XY_geometry(self, material_name, g):
        mi_drawpolygon(np.array([[0, self.D + g],
                                 [0, self.D + g + self.It],
                                 [self.A, self.D + g + self.It],
                                 [self.A, self.D + g]]))
        femm.mi_addblocklabel(self.A / 2, (self.D + g + self.It / 2))
        femm.mi_selectlabel(self.A / 2, (self.D + g + self.It / 2))
        femm.mi_setblockprop(material_name, 0, 0.5, 0, 0, 0, 0)


class PCB:
    def __init__(self, hcu):
        self.t_copper = hcu
        self.d_layers = np.array([0.3, 0.3, 0.3, 0.3, 0.3]) * 1e-3
        self.d_core = 0.2e-3


class Primary:
    def __init__(self):
        d_core_p = 1e-3  # distance core to primary
        cr_winding = 0.3e-3  # creepage between 2 traces
        self.id = 1
        self.name = 'HV'
        self.layers = [1, 2, 3]
        self.n_turn_layer = [1, 1, 1]
        self.distance_turn = cr_winding
        self.d_core_leg = d_core_p
        self.parallel = 1
        self.half = 0


class Secondary:
    def __init__(self):
        d_core_s = 1e-3  # distance core to secondary
        cr_winding = 0.3e-3  # creepage between 2 traces
        self.id = 2
        self.name = 'LV'
        self.layers = [4, 5, 6]
        self.n_turn_layer = [1, 1, 1]
        self.distance_turn = cr_winding
        self.d_core_leg = d_core_s
        self.parallel = 1
        self.half = 0


class Material:
    def __init__(self):
        # material for magnetics problems
        # mi_addmaterial(’matname’, mu_x, mu_y, Hc, J, Cduct, Lam_d, Phi_hmax, Lam_fill, LamType, Phi_hx, Phi_hy, nstr, dwire)
        # adds a new material with called ’matname’ with the material properties:
        # – mu x Relative permeability in the x- or r-direction.
        # – mu y Relative permeability in the y- or z-direction.
        # – H c Permanent magnet coercivity in Amps/Meter.
        # – J Applied source current density in Amps/mm2.
        # – Cduct Electrical conductivity of the material in MS/m.
        # – Lam d Lamination thickness in millimeters.
        # – Phi hmax Hysteresis lag angle in degrees, used for nonlinear BH curves.
        # – Lam fill Fraction of the volume occupied per lamination that is actually filled with
        # iron (Note that this parameter defaults to 1 in the femm preprocessor dialog box because,
        # by default, iron completely fills the volume)
        # – Lamtype Set to
        # 0 – Not laminated or laminated in plane
        # 1 – laminated x or r
        # 2 – laminated y or z
        # 3 – magnet wire
        # 4 – plain stranded wire
        # 5 – Litz wire
        # 6 – square wire
        # – Phi hx Hysteresis lag in degrees in the x-direction for linear problems.
        # – Phi hy Hysteresis lag in degrees in the y-direction for linear problems.
        # – nstr Number of strands in the wire build. Should be 1 for Magnet or Square wire.
        # – dwire Diameter of each of the wire’s constituent strand in millimeters.
        self.name = None
        self.mu_x = None
        self.mu_y = None
        self.perm_magn_coerc = None
        self.current_density = None
        self.sigma = None
        self.lam_thickness = None
        self.hyst_lag_angle = None
        self.fill_factor_lamination = None
        self.lamination_type = None
        self.hyst_lag_x = None
        self.hyst_lag_y = None
        self.nstrand = None
        self.dstrand = None

        self.bdata = []
        self.hdata = []

        # TODO: material for electrostatics problem
        # ei addmaterial(’matname’, ex, ey, qv) adds a new material with called ’matname’
        # with the material properties:
        # ex Relative permittivity in the x- or r-direction.
        # ey Relative permittivity in the y- or z-direction.
        # qv Volume charge density in units of C/m3

        # TODO: material for heat flow problem
        # hi addmaterial("materialname", kx, ky, qv, kt) adds a new material with called
        # "materialname" with the material properties:
        # kx Thermal conductivity in the x- or r-direction.
        # ky Thermal conductivity in the y- or z-direction.
        # qv Volume heat generation density in units of W/m3
        # kt Volumetric heat capacity in units of MJ/(m3*K)

        # material for current flow problems
        # ci addmaterial("materialname", ox, oy, ex, ey, ltx, lty) adds a new material
        # with called "materialname" with the material properties:
        # ox Electrical conductivity in the x- or r-direction in units of S/m
        # oy Electrical conductivity in the y- or z-direction in units of S/m
        # ex Relative permittivity in the x- or r-direction
        # ey Relative permittivity in the y- or z-direction
        # ltx Dielectric loss tangent in the x- or r-direction
        # lty Dielectric loss tangent in the y- or z-direction
        self.sigma_x = None
        self.sigma_y = None
        self.epsi_x = None
        self.epsi_y = None
        self.tand_x = None
        self.tand_y = None

    def set_magnetic_material(self, materialname: str, mu_x: float, mu_y: float, Hc: float, J: float, Cduct: float,
                              Lam_d: float, Phi_hmax: float, Lam_fill: float, LamType: int, Phi_hx: float,
                              Phi_hy: float, nstr: int, dwire: float):
        self.name = materialname
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.perm_magn_coerc = Hc
        self.current_density = J
        self.sigma = Cduct
        self.lam_thickness = Lam_d
        self.hyst_lag_angle = Phi_hmax
        self.fill_factor_lamination = Lam_fill
        self.lamination_type = LamType
        self.hyst_lag_x = Phi_hx
        self.hyst_lag_y = Phi_hy
        self.nstrand = nstr
        self.dstrand = dwire
        self.bdata = []
        self.hdata = []


class Air(Material):
    def __init__(self):
        self.name = 'Air'
        self.mu_x = 1
        self.mu_y = 1
        self.perm_magn_coerc = 0
        self.current_density = 0
        self.sigma = 0
        self.lam_thickness = 0
        self.hyst_lag_angle = 0
        self.fill_factor_lamination = 1
        self.lamination_type = 0
        self.hyst_lag_x = 0
        self.hyst_lag_y = 0


class LinearIron(Material):
    def __init__(self):
        self.name = 'Air'
        self.mu_x = 2100
        self.mu_y = 2100
        self.perm_magn_coerc = 0
        self.current_density = 0
        self.sigma = 0
        self.lam_thickness = 0
        self.hyst_lag_angle = 0
        self.fill_factor_lamination = 1
        self.lamination_type = 0
        self.hyst_lag_x = 0
        self.hyst_lag_y = 0


class NotLinearIron(Material):
    def __init__(self):
        self.name = 'Air'
        self.mu_x = 2100
        self.mu_y = 2100
        self.perm_magn_coerc = 0
        self.current_density = 0
        self.sigma = 0
        self.lam_thickness = 0
        self.hyst_lag_angle = 0
        self.fill_factor_lamination = 1
        self.lamination_type = 0
        self.hyst_lag_x = 0
        self.hyst_lag_y = 0
        # A set of points defining the BH curve is then specified.
        self.bdata = [0., 0.3, 0.8, 1.12, 1.32, 1.46, 1.54, 1.62, 1.74, 1.87, 1.99, 2.046, 2.08]
        self.hdata = [0, 40, 80, 160, 318, 796, 1590, 3380, 7960, 15900, 31800, 55100, 79600]


class Ferroxcube_3C95(Material):
    def __init__(self):
        self.name = 'Ferroxcube_3C95'
        self.mu_x = 3000
        self.mu_y = 3000
        # self.permeability = [3000, 3000]  # [mur_x mur_y] Relative permeability in the x and y direction.
        self.sigma = 0.2e-6


# class Polygon:
#     def __init__(self):
#         self.name = 'foo'
#         self.x = np.array()  # list of x coord of outer points
#         self.y = np.array()  # list of y coord of outer points
#         # TODO: avoid this workaround of dx,dy using algorithm to find internal points see
#         #  https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
#         self.dx = 0.0  # resolution on x axis (used to place label inside the geometry)
#         self.dy = 0.0  # resolution on x axis (used to place label inside the geometry)
#
#     def create_polygon(self, name: str, x_E: np.array, y_E: np.array, x_res: float, y_res: float):
#         self.name = name
#         self.x = x_E  # list of x coord of outer points
#         self.y = y_E  # list of y coord of outer points
#         # TODO: avoid this workaround of dx,dy using algorithm to find internal points see
#         #  https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
#         self.dx = x_res  # resolution on x axis (used to place label inside the geometry)
#         self.dy = y_res  # resolution on x axis (used to place label inside the geometry)
#
#     def draw_polygon_femm(self):
#         mi_drawpolygon([self.x, self.y])


class MagneticCircuit:
    def __init__(self, circuit_name, current, circuittype):
        self.name = circuit_name
        self.current = current
        self.circuit_type = circuittype


class Label:
    def __init__(self):
        self.name = None
        self.xpos = None
        self.ypos = None


class Block:
    def __init__(self):
        self.name = 'foo'
        self.geometry = None
        self.material = Material()
        self.excitation = MagneticCircuit()
        self.label = Label()

    def set_label(self, name: str):
        self.label.xpos = min(self.geometry.x) + self.geometry.dx
        self.label.ypos = min(self.geometry.y) + self.geometry.dy

    def add_label_femm(self):
        femm.mi_addblocklabel(self.label.xpos, self.label.ypos)

    def select_label_femm(self):
        femm.mi_selectlabel(self.label.xpos, self.label.xpos)

    def add_material_femm(self, m: Material):
        if self.problem.solver == 'magnetic':
            femm.mi_addmaterial(m.name,
                                m.mu_x,
                                m.mu_y,
                                m.perm_magn_coerc,
                                m.current_density,
                                m.sigma,
                                m.lam_thickness,
                                m.hyst_lag_angle,
                                m.fill_factor_lamination,
                                m.lamination_type,
                                m.hyst_lag_x,
                                m.hyst_lag_y,
                                m.nstrand,
                                m.dstrand)

    def add_non_linear_material_femm(self, m: Material):
        if self.problem.solver == 'magnetic':
            femm.mi_addmaterial(m.name,
                                m.mu_x,
                                m.mu_y,
                                m.perm_magn_coerc,
                                m.current_density,
                                m.sigma,
                                m.lam_thickness,
                                m.hyst_lag_angle,
                                m.fill_factor_lamination,
                                m.lamination_type,
                                m.hyst_lag_x,
                                m.hyst_lag_y,
                                m.nstrand,
                                m.dstrand)
            for item in range(0, len(m.bdata)):
                femm.mi_addbhpoint(m.name, m.bdata[item], m.hdata[item])


class FEMMMagneticProblem:
    def __init__(self):
        # load default values
        # - freq to the desired frequency in Hertz for AC simulation
        # - units parameter specifies the units used for measuring length in the problem domain.
        # - Valid ’units’ entries are ’inches’, ’millimeters’, ’centimeters’, ’mils’, ’meters’, and ’micrometers’.
        # - Set the parameter problemtype to ’planar’ for a 2-D planar problem, or to ’axi’ for an axisymmetric problem.
        # - The precision parameter dictates the precision required by the solver. For example, entering 1E-8 requires the RMS of the residual to be less than 10−8.
        # - representing the depth of the problem in the into-the-page direction for 2-D planar problems. Specify the depth to be zero for axisymmetric problems.
        # - the minimum angle constraint sent to the mesh generator – 30 degrees is the usual choice for this parameter.
        # - The acsolver parameter specifies which solver is to be used for AC problems: 0 for successive approximation, 1

        self.solver = 'magnetic'
        self.frequency = 0
        self.units = 'meters'
        self.problem_type = 'axi'  # axial problem
        self.precision = 1.e-8
        self.planar_problem_depth = 0
        self.acsolver = 0  # successive approximations

class FEMM:
    def __init__(self):
        self.filename = "pippo"
        self.problem = FEMMMagneticProblem()
        self.blocks_list = None

    def open(self):
        # The package must be initialized with the openfemm command.
        femm.openfemm()

    def new_magnetic_planar_problem(self):
        # We need to create a new Magnetostatics document to work on.
        femm.newdocument(0)
        # mi_probdef(freq, units, type, precision, depth, minangle, (acsolver)) changes the problem definition.
        self.problem = FEMMMagneticProblem()
        self.problem.frequency = 0
        self.problem.problem_type = 'planar'
        femm.mi_probdef(self.problem.frequency, self.problem.units, self.problem.problem_type, self.problem.precision,
                        self.problem.planar_problem_depth, self.problem.acsolver)

    def set_frequency(self, frequency: float):
        self.problem.frequency = frequency
        femm.mi_probdef(self.problem.frequency, self.problem.units, self.problem.problem_type, self.problem.precision,
                        self.problem.planar_problem_depth, self.problem.acsolver)

    def set_depth(self, z_depth: float):
        self.problem.planar_problem_depth = z_depth
        femm.mi_probdef(self.problem.frequency, self.problem.units, self.problem.problem_type, self.problem.precision,
                        self.problem.planar_problem_depth, self.problem.acsolver)

    def set_open_boundary_condition(self):
        if self.problem.solver == 'magnetic':
            # Define an "open" boundary condition using the built-in function:
            femm.mi_makeABC()

    def add_block_label(self, x, y):
        if self.problem.solver == 'magnetic':
            # Add block labels, one to each the steel, coil, and air regions.
            femm.mi_addblocklabel(x, y)

    def add_circuit_propriety(self, block: Block):
        if self.problem.solver == 'magnetic':
            # mi_addcircprop(’circuitname’, i, circuittype) adds a new circuit property with name ’circuitname’ with
            # a prescribed current.The circuittype parameter is 0 for a parallel-connected circuit
            # and 1 for a series-connected circuit.
            femm.mi_addcircprop(Block.circuit_name, Block.current, Block.circuittype)

    def set_block_material(self, block: Block):
        if self.problem.solver == 'magnetic':
            if block.material == 'linear':
                femm.mi_selectlabel(5, 0)
                femm.mi_setblockprop('Iron', 0, 1, '<None>', 0, 0, 0)
                femm.mi_clearselected()

    def zoom_natural(self):
        if self.problem.solver == 'magnetic':
            # Now, the finished input geometry can be displayed.
            femm.mi_zoomnatural()

    def save_file(self):
        if self.problem.solver == 'magnetic':
            # We have to give the geometry a name before we can analyze it.
            femm.mi_saveas(self.filename)

    def save_file_copy(self, filename):
        if self.problem.solver == 'magnetic':
            # We have to give the geometry a name before we can analyze it.
            femm.mi_saveas(filename)

    def get_magnetic_flux_xy(self, x, y):
        if self.problem.solver == 'magnetic':
            # If we were interested in the flux density at specific positions,
            # we could inquire at specific points directly:
            flux = femm.mo_getb(x, y)
            return flux[1]

    def get_electrical_param(self, circuit: MagneticCircuit):
        if self.problem.solver == 'magnetic':
            # The program will report the terminal properties of the circuit:
            # current, voltage, and flux linkage
            [current, voltage, phi] = femm.mo_getcircuitproperties(circuit.name)
            return [current, voltage, phi]

    def get_inductance(self, circuit: MagneticCircuit):
        if self.problem.solver == 'magnetic':
            # If we were interested in inductance, it could be obtained by
            # dividing flux linkage by current
            [current, voltage, phi] = femm.mo_getcircuitproperties(circuit.name)
            inductance = phi / current
            return inductance

    def get_flux_along_path(self, x_path: list, y_path: list):
        if self.problem.solver == 'magnetic':
            # Or we could, for example, plot the results along a path
            flux_path = []
            for x, y in zip(x_path, y_path):
                flux = femm.mo_getb(x, y)
                flux_path.append(flux[1])
            return flux_path

    def solve(self):
        if self.problem.solver == 'magnetic':
            # Now,analyze the problem and load the solution when the analysis is finished
            femm.mi_analyze()
            femm.mi_loadsolution()

    def close(self):
        femm.closefemm()

    def draw_blocks(self):
        # Use either automesh or minmesh
        automesh = 1
        minmesh = 0.5  # minimum mesh size in mm(No effect if automesh = 1)

        for item in self.blocks_list:
            item.geometry.draw_femm()
            item.set_material_femm(item.material)
            femm.mi_setblockprop(item.material.name, automesh, minmesh, 0, 0, item, 0)


    def CreatePolyline(self, polyline_array: np.array):
        mi_drawpolygon(polyline_array)


def create_Ecore(core: EcoreGeometry, material: Material, gap):
    mi_drawpolygon(np.array([[0, 0], [0, core.D],
                             [(core.A - core.B) / 2, core.D],
                             [(core.A - core.B) / 2, core.D - core.E],
                             [(core.A - core.C) / 2, core.D - core.E],
                             [(core.A - core.C) / 2, core.D],
                             [(core.A + core.C) / 2, core.D],
                             [(core.A + core.C) / 2, core.D - core.E],
                             [(core.A + core.C) / 2, core.D - core.E],
                             [(core.A + core.B) / 2, core.D - core.E],
                             [(core.A + core.B) / 2, core.D],
                             [core.A, core.D], [core.A, 0]]))
    femm.mi_addblocklabel(core.A / 2, core.D / 2)
    femm.mi_selectlabel(core.A / 2, core.D / 2)
    femm.mi_setblockprop(material.name, 0, 0.5, 0, 0, 0, 0)


def drawEEcoreGapEachLeg(core: EcoreGeometry, material: Material, gap):
    # define the mu_x and mu_y
    femm.mi_addmaterial(material.name, material.mu_x, material.mu_y)
    femm.mi_modifymaterial(material.name, 5, material.sigma)
    # here we design the 2D of the E core, see PlanarFEMM_diagram.png for more
    create_Ecore(core, materia, gap / 2)
    femm.mi_selectgroup(0)
    # in this case we suppose the gap is on all of the core legs
    femm.mi_mirror(0, core.D, core.A, core.D)


def drawEIcoreGapEachLeg(core: EcoreGeometry, material: Material, t, gap):
    # define the mu_x and mu_y
    femm.mi_addmaterial(material.name, material.mu_x, material.mu_y)
    # store the ferrite conductivity
    femm.mi_modifymaterial(material.name, 5, material.sigma)
    # draw Ecore with internal leg shortened by defined gap
    create_Ecore(core, material, 0)
    create_Icore(core, t, material, gap / 2)


def drawEEcoreGapMidLeg(Core, material, gap):
    femm.mi_addmaterial(material.name, material.mu_x, material.mu_y)
    femm.mi_modifymaterial(material.name, 5, material.sigma)
    # here we design the 2D of the E core, see PlanarFEMM_diagram.png for more
    create_Ecore(Core, material, gap / 2)
    femm.mi_addblocklabel(Core.A / 2, Core.D / 2)
    femm.mi_selectlabel(Core.A / 2, Core.D / 2)
    # assign the material to the label just designed
    femm.mi_setblockprop(material.name, 0, 0.5, 0, 0, 0, 0)
    femm.mi_selectgroup(0)
    # in this case we suppose the gap is on all of the core legs
    femm.mi_mirror(0, Core.D, Core.A, Core.D)


# def create_Ecore(core, material_name, gap):
#     c = core
#     mi_drawpolygon(np.array([[0, 0], [0, c.D], [(c.A - c.B) / 2, c.D], [(c.A - c.B) / 2, c.D - c.E],
#                              [(c.A - c.C) / 2, c.D - c.E], [(c.A - c.C) / 2, c.D], [(c.A + c.C) / 2, c.D],
#                              [(c.A + c.C) / 2, c.D - c.E], [(c.A + c.C) / 2, c.D - c.E],
#                              [(c.A + c.B) / 2, c.D - c.E], [(c.A + c.B) / 2, c.D],
#                              [c.A, c.D], [c.A, 0]]))
#     femm.mi_addblocklabel(c.A / 2, c.D / 2)
#     femm.mi_selectlabel(c.A / 2, c.D / 2)
#     femm.mi_setblockprop(material_name, 0, 0.5, 0, 0, 0, 0)


def create_Icore(core, t, material: Material, g):
    c = core
    pts = np.array([[0, c.D + g], [0, c.D + g + t], [c.A, c.D + g + t], [c.A, c.D + g]])
    mi_drawpolygon(pts)
    femm.mi_addblocklabel(c.A / 2, (c.D + g + t / 2))
    femm.mi_selectlabel(c.A / 2, (c.D + g + t / 2))
    femm.mi_setblockprop(material.name, 0, 0.5, 0, 0, 0, 0)


def drawRectWithCenterWidthHeight(xc, yc, w, h, material: Material):
    pts = np.array(
        [[xc - w / 2, yc - h / 2], [xc - w / 2, yc + h / 2], [xc + w / 2, yc + h / 2], [xc + w / 2, yc - h / 2]])
    mi_drawpolygon(pts)
    femm.mi_addblocklabel(xc, yc)
    femm.mi_selectlabel(xc, yc)
    femm.mi_setblockprop(material.name, 0, 0.5, 0, 0, 0, 0)


def create_EEcore(Core, material: Material, g):
    femm.mi_addmaterial(material.name, material.mu_x, material.mu_y)
    femm.mi_modifymaterial(material.name, 5, material.sigma)
    create_Ecore(Core, material, 0)
    femm.mi_selectgroup(0)
    # in this case we suppose the gap is on all of the core legs
    femm.mi_mirror(0, Core.D + g / 2, Core.A, Core.D + g / 2)


def create_EIcore(Core, t, material: Material, g):
    femm.mi_addmaterial(material.name, material.mu_x, material.mu_y)
    femm.mi_modifymaterial(material.name, 5, material.sigma)
    create_Ecore(Core, material)
    create_Icore(Core, t, material, g)

def set_circuit(circ, layers, nturn_layer, I):
    h = 1
    for i in range(len(layers)):
        for j in range(int(np.ceil(nturn_layer[i]))):
            str_pos = str(h) + circ + '+'
            str_neg = str(h) + circ + '-'
            femm.mi_modifycircprop(str_pos, 1, I * (1 - 0.5 * (j > nturn_layer[i])))
            femm.mi_modifycircprop(str_neg, 1, -I * (1 - 0.5 * (j > nturn_layer[i])))
            h = h + 1


def set_circuit_half(circ, layers, nturn_layer, I):
    h = 1
    for i in range(len(layers)):
        str_pos = str(h) + circ + '+'
        str_neg = str(h) + circ + '-'
        str_ext_pos = ['ext' + str(h) + circ + '+']
        str_ext_neg = [str(h) + circ + '-']
        femm.mi_modifycircprop(str_pos, 1, I / 2)
        femm.mi_modifycircprop(str_ext_pos, 1, -I / 2)
        femm.mi_modifycircprop(str_neg, 1, -I / 2)
        femm.mi_modifycircprop(str_ext_neg, 1, I / 2)
        h = h + 1



def test_open(Core, winding, id, PCB, g):
    femm.mi_getmaterial('Air')

    if g == 0:
        femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

        femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_probdef(0, 'meters', 'planar', 1e-8, Core.F)
    femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)

    femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_open.fem']))
    femm.mi_createmesh()

    for i in range(len(winding)):
        if i == id:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
        else:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)

    femm.mi_analyze(1)
    femm.mi_loadsolution()
    femm.mo_groupselectblock(id)
    Rdc = np.real(femm.mo_blockintegral(4))
    femm.mo_clearblock()
    femm.mo_groupselectblock(0)
    femm.mo_groupselectblock(99)
    Lself = 2 * np.real(femm.mo_blockintegral(2))
    femm.mo_clearblock()

    # reset currents to 0
    for i in range(len(winding)):
        if winding[i].half:
            set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
        else:
            set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)

    return [Rdc, Lself]


def test_power(Core, winding, id, f, PCB, I, g):
    femm.mi_getmaterial('Air')

    if g == 0:
        femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

        femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_probdef(f, 'meters', 'planar', 1e-8, Core.F)
    femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)

    femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_power.fem']))
    femm.mi_createmesh()

    n = []
    Rac = []
    for i in range(len(winding)):
        n[i] = np.sum(winding[id - 1].n_turn_layer) / winding[id - 1].parallel / np.sum(winding[i].n_turn_layer) * \
               winding[i].parallel
        if i == id:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, I / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, I / winding[i].parallel)

        else:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer,
                                 -n[i] * I / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer,
                            -n[i] * I / winding[i].parallel)

    femm.mi_analyze(1)
    femm.mi_loadsolution()
    for i in range(len(winding)):
        if i == id:
            femm.mo_groupselectbloc(i)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4))
            femm.mo_clearblock()
        else:
            femm.mo_groupselectblock(i)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4)) / n[i] ^ 2
            femm.mo_clearblock()

    femm.mo_groupselectblock(0)
    femm.mo_groupselectblock(99)
    L_leak = 4 * np.real(femm.mo_blockintegral(2))
    femm.mo_clearblock()

    femm.mo_resize(800, 800)
    femm.mo_zoomnatural()
    femm.mo_zoomin()
    femm.mo_savebitmap(''.join([os.getcwd(), '\\img\\test_short.bmp']))
    femm.mo_savemetafile(''.join([os.getcwd(), '\\img\\test_short.eps']))
    # reset currents to 0
    for i in range(len(winding)):
        if winding[i].half:
            set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
        else:
            set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)

    return Rac, L_leak


def test_saturation(Core, winding, id, PCB, I, g):
    femm.mi_getmaterial('Air')

    if g == 0:
        femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

        femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_probdef(0, 'meters', 'planar', 1e-8, Core.F)
    femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)

    femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_saturation.fem']))
    femm.mi_createmesh()

    for i in range(len(winding)):
        if i == id:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, I / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, I / winding[i].parallel)
        else:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)

    femm.mi_analyze(1)
    femm.mi_loadsolution()()
    femm.mo_clearblock()

    # reset currents to 0
    for i in range(len(winding)):
        if winding[i].half:
            set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
        else:
            set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)





def test_short(Core, winding, id, f, PCB, g):
    femm.mi_getmaterial('Air')

    if g == 0:
        femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

        femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_probdef(f, 'meters', 'planar', 1e-8, Core.F)
    femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)

    femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_short.fem']))
    femm.mi_createmesh()

    n = np.zeros(shape=np.shape(winding))
    Rac = np.zeros(shape=np.shape(winding))

    for i in range(len(winding)):
        n[i] = np.sum(winding[id].n_turn_layer) / winding[id].parallel / np.sum(winding[i].n_turn_layer) * winding[
            i].parallel
        if i == id - 1:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
        else:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer,
                                 -n[i] / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, -n[i] / winding[i].parallel)

    femm.mi_analyze(1)
    femm.mi_loadsolution()
    for i in range(len(winding)):
        if i == id:
            femm.mo_groupselectblock(i)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4))
            femm.mo_clearblock()
        else:
            femm.mo_groupselectblock(i)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4)) / n[i] ** 2
            femm.mo_clearblock()

    femm.mo_groupselectblock(0)
    femm.mo_groupselectblock(99)
    L_leak = 4 * np.real(femm.mo_blockintegral(2))
    femm.mo_clearblock()

    femm.mo_resize(800, 800)
    femm.mo_zoomnatural()
    femm.mo_zoomin()
    femm.mo_savebitmap(''.join([os.getcwd(), '\\img\\test_short.bmp']))
    femm.mo_savemetafile(''.join([os.getcwd(), '\\img\\test_short.eps']))
    # reset currents to 0
    for i in range(len(winding)):
        if winding[i].half:
            set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
        else:
            set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)

    return [Rac, L_leak]


def test_short_CT(Core, winding, id, f, PCB, g):
    femm.mi_getmaterial('Air')

    if g == 0:
        femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

        femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
        femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
    femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)

    femm.mi_probdef(f, 'meters', 'planar', 1e-8, Core.F)
    femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)

    femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_short.fem']))
    femm.mi_createmesh()

    n = []
    Rac = []

    for i in range(len(winding)):
        n[i] = np.sum(winding[id - 1].n_turn_layer) / winding[id - 1].parallel / np.sum(winding[i].n_turn_layer) * \
               winding[i].parallel
        if i == id:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
        else:
            if winding[i].half:
                set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer,
                                 -n[i] / winding[i].parallel)
            else:
                set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, -n[i] / winding[i].parallel)

    femm.mi_analyze(1)
    femm.femm.mi_loadsolution()
    for i in range(len(winding)):
        if i == id:
            femm.mo_groupselectblock(i)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4))
            femm.mo_clearblock()
        else:
            femm.mo_groupselectblock(i)
            femm.mo_groupselectblock(i + 1)
            Rac[i] = 2 * np.real(femm.mo_blockintegral(4)) / n[i] ^ 2
            femm.mo_clearblock()

    femm.mo_groupselectblock(0)
    femm.mo_groupselectblock(99)
    L_leak = 4 * np.real(femm.mo_blockintegral(2))
    femm.mo_clearblock()

    femm.mo_resize(800, 800)
    femm.mo_zoomnatural()
    femm.mo_zoomin()
    femm.mo_savebitmap(''.join([os.getcwd(), '\\img\\test_short.bmp']))
    femm.mo_savemetafile(''.join([os.getcwd(), '\\img\\test_short.eps']))
    # reset currents to 0
    for i in range(len(winding)):
        if winding[i].half:
            set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
        else:
            set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
    return [Rac, L_leak]


def mi_drawpolyline(p):
    for k in range(0, len(p) - 1):
        femm.mi_drawline(p[k, 0], p[k, 1], p[k + 1, 0], p[k + 1, 1])


def mi_drawrectangle(x1, y1, x2, y2):
    femm.mi_drawline(x1, y1, x2, y1)
    femm.mi_drawline(x2, y1, x2, y2)
    femm.mi_drawline(x2, y2, x1, y2)
    femm.mi_drawline(x1, y2, x1, y1)


def mi_drawpolygon(p: np.array):
    mi_drawpolyline(p)
    n = len(p)
    femm.mi_drawline(p[0, 0], p[0, 1], p[n - 1, 0], p[n - 1, 1])


def drawEwinding_halfT(Core, PCB, winding, material_name):
    # This funtion draws the half turn planar windings in FEMM
    femm.mi_getmaterial(material_name)

    id = winding.id
    layers = winding.layers
    d_core_leg = winding.d_core_leg
    nturn_layer = winding.n_turn_layer
    distance_turn = winding.distance_turn
    circ = winding.name

    d_core_pcb = PCB.d_core
    d_layers = np.hstack((0, PCB.d_layers))
    t_copper = PCB.t_copper

    c = Core

    h = 1
    for i in np.arange(layers):
        turn_size = (c.B - c.C - 4 * d_core_leg) / 2
        for j in np.ceil(nturn_layer[i]):
            str_pos = str(h) + circ + '+'
            str_neg = str(h) + circ + '-'
            str_ext_pos = 'ext' + str(h) + circ + '+'
            str_ext_neg = str(h) + circ + '-'

            femm.mi_addcircprop(str_pos, 0, 1)
            femm.mi_addcircprop(str_ext_pos, 0, 1)
            femm.mi_addcircprop(str_neg, 0, 1)
            femm.mi_addcircprop(str_ext_neg, 0, 1)
            if j == 0:
                x1 = ((c.A - c.B) / 2) + d_core_leg
            else:
                x1 = x1 + distance_turn + turn_size

            if layers[i] == 1:
                y1 = (c.D - c.E) + d_core_pcb
            else:
                y1 = (c.D - c.E) + d_core_pcb + np.sum(d_layers[0:layers[i]]) + np.sum(
                    t_copper[0: layers[i]])  # *(layers[i]-1))

            x2 = x1 + turn_size
            y2 = y1 + t_copper[layers[i] - 1]  # +t_copper(len(layers))

            femm.mi_drawrectangle(x1, y1, x2, y2)
            femm.mi_addblocklabel((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
            femm.mi_selectlabel((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
            femm.mi_setblockprop(material_name, 0, 0.05, str_pos, 0, id, 0)

            femm.mi_selectrectangle(x1, y1, x2, y2)
            femm.mi_mirror((c.A - c.B) / 4, 0, (c.A - c.B) / 4, c.D)
            femm.mi_selectlabel((c.A - c.B) / 4 - ((x2 - x1) / 2 + x1), (y2 - y1) / 2 + y1)
            femm.mi_setblockprop(material_name, 0, 0.05, str_ext_pos, 0, id, 0)

            femm.mi_selectrectangle(x1, y1, x2, y2)
            femm.mi_mirror(c.A / 2, 0, c.A / 2, c.D)
            femm.mi_selectlabel(c.A - ((x2 - x1) / 2 + x1), (y2 - y1) / 2 + y1)
            femm.mi_setblockprop(material_name, 0, 0.05, str_neg, 0, id, 0)

            femm.mi_selectrectangle((c.A - c.B) / 2 - x1, y1, (c.A - c.B) / 2 - x2, y2)
            femm.mi_mirror(c.A / 2, 0, c.A / 2, c.D)
            femm.mi_selectlabel((3 * c.A + c.B) / 4 + ((x2 - x1) / 2 + x1), (y2 - y1) / 2 + y1)
            femm.mi_setblockprop(material_name, 0, 0.05, str_ext_neg, 0, id, 0)

            h = h + 1

        femm.mi_clearselected()


def drawEwinding(Core, PCB, winding, material_name):
    # This funtion draws the planar windings in FEMM
    femm.mi_getmaterial(material_name)

    id = winding.id
    layers = winding.layers
    d_core_leg = winding.d_core_leg
    nturn_layer = winding.n_turn_layer
    distance_turn = winding.distance_turn
    circ = winding.name

    d_core_pcb = PCB.d_core
    d_layers = np.hstack((0, PCB.d_layers))
    t_copper = PCB.t_copper

    c = Core

    h = 1
    for i in range(len(layers)):
        turn_size = (c.B - c.C - 4 * d_core_leg - 2 * (np.ceil(nturn_layer[i]) - 1) * distance_turn) / 2 / \
                    nturn_layer[i]
        # Distance between each turn
        for j in range(int(np.ceil(nturn_layer[i]))):
            str_pos = str(h) + circ + '+'
            str_neg = str(h) + circ + '-'

            femm.mi_addcircprop(str_pos, 0, 1)
            femm.mi_addcircprop(str_neg, 0, 1)

            if j == 0:
                x1 = ((c.A - c.B) / 2) + d_core_leg
            else:
                x1 = x1 + distance_turn + turn_size

            if layers[i] == 0:
                y1 = (c.D - c.E) + d_core_pcb
            else:
                y1 = (c.D - c.E) + d_core_pcb + np.sum(d_layers[1:layers[i]]) + np.sum(
                    t_copper[1: layers[i]])  # *(layers[i]-1))

                x2 = x1 + turn_size * (1 - 0.5 * (j > nturn_layer[i]))
                y2 = y1 + t_copper[layers[i] - 1]  # +t_copper(len(layers))
                femm.mi_drawrectangle(x1, y1, x2, y2)
                femm.mi_addblocklabel((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
                femm.mi_selectlabel((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
                femm.mi_setblockprop(material_name, 0, 0.05, str_pos, 0, id, 0)
                femm.mi_selectgroup(id)
                femm.mi_mirror(c.A / 2, 0, c.A / 2, c.D)
                femm.mi_selectlabel(c.A - ((x2 - x1) / 2 + x1), (y2 - y1) / 2 + y1)
                femm.mi_setblockprop(material_name, 0, 0.05, str_neg, 0, id, 0)
                femm.mi_selectrectangle(x1, y1, x2, y2)
                femm.mi_mirror(c.A / 2, 0, c.A / 2, c.D)
                h = h + 1


def getCoordEIcore(Core):
    # Outputs vectors [x,y] of the coordinate points that define the cores as polygons
    # [x_E] = E core
    # [x_I] = I core

    c = Core

    # E-core
    xy_E = [[0, 0],
            [0, c.D],
            [(c.A - c.B) / 2, c.D],
            [(c.A - c.B) / 2, (c.D - c.E)],
            [(c.A - c.C) / 2, (c.D - c.E)],
            [(c.A - c.C) / 2, c.D],
            [(c.A + c.C) / 2, c.D],
            [(c.A + c.C) / 2, (c.D - c.E)],
            [(c.A + c.C) / 2, (c.D - c.E)],
            [(c.A + c.B) / 2, (c.D - c.E)],
            [(c.A + c.B) / 2, (c.D)],
            [c.A, c.D],
            [c.A, 0]]
    x_E = xy_E[:, 1]
    y_E = xy_E[:, 2]

    # Icore
    xy_I = [[0, (c.D + c.g)],
            [0, (c.D + c.g + c.It)],
            [(c.A), (c.D + c.g + c.It)],
            [(c.A), (c.D + c.g)]]

    x_I = xy_I[:, 1]
    y_I = xy_I[:, 2]

    return [x_E, y_E, x_I, y_I]


def test_trafo_E22_6_16():
    ## constant definition
    sigmaCu_20 = 6e7  # copper conductivity at 20�C
    mu0 = 4 * np.pi * 1e-7

    ## electrical paramaters
    n_HV = 3  # Primary number of turns
    n_LV = 3
    fs = 200e3

    ## winding stack description
    n_layer = 6

    hcu = np.array([105e-6, 105e-6, 105e-6, 105e-6, 105e-6, 105e-6])  # copper layer thickness
    # total copper foil thickness of all layers
    tlp = np.array([1, 1, 1, 0, 0, 0])  # description of turn of primary in layer stack ex 2T in 2nd layer
    np_HV = 1  # number of turn in parallel for the primary
    tls = np.array([0, 0, 0, 1, 1, 1])
    np_LV = 1  # number of turn in parallel for the sec

    d_core_p = 1e-3  # distance core to primary
    d_core_s = 1e-3  # distance core to secondary
    cr_winding = 0.3e-3  # creepage between 2 traces

    ## core description
    # core dimensions as defined in PlanarFEMM_diagram.png
    # for E22/6/16 in E/PLT arrangement
    # https://www.ferroxcube.com/en-global/download/download/11
    # Dimension defined as in PlanarFEMM_diagram.png (NOTE THAT planarFEMM TAKES DIMENSIONS IN MM)
    E22616 = EcoreGeometry(A=21.8e-3, B=16.8e-3, C=5e-3, D=5.7e-3, E=3.2e-3, F=15.8e-3)
    PLT22616 = Icore(A=21.8e-3, F=15.8e-3, t=2.5e-3)
    E22616_material = Ferroxcube_3C95()

    myPCB = PCB(hcu=hcu)
    myPrimary = Primary()
    mySecondary = Secondary()

    # E/PLT arrangement
    ww = (E22616.A - E22616.B) / 2  # winding width
    hw = E22616.E  # winding width
    wc = E22616.C  # core width
    hc = E22616.F  # core depth

    ## calculation section
    lw = 2 * hc + 2 * wc + np.pi * ww  # wire mean path lengths
    lc = hw + ww + np.pi * wc / 2  # mean magnetic path length

    # E/E arrangement
    # lc = 2*hw + 2*ww + np.pi*wc/2 #

    # primary winding resistance
    tw_p = (ww - cr_winding * (np.ceil(tlp) - 1) - 2 * d_core_p) / tlp  # trace width in each primary layer
    Rwp = np.sum(tlp * lw / tw_p / hcu / sigmaCu_20) / np_HV ** 2  # np.sum each turn resistance
    # the same for secondary winding resistance
    w_mean_s = (ww - cr_winding * (np.ceil(tls) - 1) - 2 * d_core_s) / tls
    # the sec resistance reflected to primary
    Rws = np.sum(tls * lw / w_mean_s / hcu / sigmaCu_20) / np_LV ** 2
    Rws2p = n_HV ** 2 / np_LV ** 2 * Rws
    Rt = Rwp + Rws2p

    # open a new femm document
    femm.openfemm()
    femm.newdocument(0)
    femm.mi_probdef(0, 'meters', 'planar', 1e-8, E22616.F)
    femm.mi_makeABC(7, 2 * E22616.A, E22616.A / 2, E22616.D / 2, 0)
    gap = 0  # distributed gap length in mm

    drawEIcoreGapEachLeg(core=E22616, t=PLT22616.t, material=E22616_material, gap=gap)
    # drawEIcoreGapMidLeg(E22616, PLT22616.t, E22616_material, gap)
    # drawEEcoreGapEachLeg(core=E22616,  t=PLT22616.t, material=E22616_material, gap=gap)
    # drawEEcoreGapMidLeg(E22616, E22616, E22616_material, gap)

    drawEwinding(E22616, myPCB, myPrimary, 'Copper')
    drawEwinding(E22616, myPCB, mySecondary, 'Copper')
    time.sleep(2)

    Rdc_p, L_self_p = test_open(E22616, [myPrimary, mySecondary], myPrimary.id, myPCB, gap)
    Rdc_s, L_self_s = test_open(E22616, [myPrimary, mySecondary], myPrimary.id, myPCB, gap)

    print('Rdc_p= %e' % Rdc_p)
    print('Rdc_s= %e' % Rdc_s)

    print('L_self_p= %e' % L_self_p)
    print('L_self_s= %e' % L_self_p)

    Rac, L_leak = test_short(E22616, [myPrimary, mySecondary], myPrimary.id, fs, myPCB, gap)
    ratioRacRdc = (Rac[0] + Rac[1] * n_HV ** 2) / (Rdc_p + n_HV ** 2 * Rdc_s)
    L_leak_mm = L_leak / E22616.F

    L_leak = L_leak_mm * lw
    k = np.sqrt(1 - L_leak / L_self_p)

    # test_saturation(E22616, [myPrimary, mySecondary], myPrimary.id, myPCB, Im/np.sqrt(2), gap)


if __name__ == "__main__":
    trafo_FEM = FEMM()
    trafo_FEM.open()
    trafo_FEM.new_magnetic_planar_problem()
    trafo_FEM.set_frequency(frequency=0.0)
    test_trafo_E22_6_16()
