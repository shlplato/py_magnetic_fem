import os
import numpy as np
import femm
import time
import copy


# from shapely.geometry import Point, Polygon


# def inpolygon(xq, yq, xv, yv):
#     shape = xq.shape
#     xq = xq.reshape(-1)
#     yq = yq.reshape(-1)
#     xv = xv.reshape(-1)
#     yv = yv.reshape(-1)
#     q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
#     p = mpl.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
#     in_pts = p.contains_points(q).reshape(shape)
#     on_pts = None  # TODO:
#     return in_pts, on_pts


# Finds a point 'X,Y' inside each polygon defined in 'blocks'
def find_in_pts_in_block(block):
    # labelXY = []
    #
    # # Test center of polygon
    # xtest = (np.max(block.geometry.x) + np.min(block.geometry.x)) / 2
    # ytest = (np.max(block.geometry.y) + np.min(block.geometry.y)) / 2
    #
    # [in_pts, on_pts] = inpolygon(xtest, ytest, block.geometry.x, block.geometry.y)
    #
    # if in_pts == 1 and on_pts == 0:
    #     labelXY[0, :] = [xtest, ytest]
    # else:
    #     xtest = sum(block.geometry.x) / len(block.geometry.x)
    #     ytest = sum(block.geometry.y) / len(block.geometry.y)
    #     [in_pts, on_pts] = inpolygon(xtest, ytest, block.geometry.x, block.geometry.y)
    #
    #     if in_pts == 1 and on_pts == 0:
    #         labelXY[0, :] = [xtest, ytest]
    #     else:
    #         # Test edges of polygon
    #         for j in range(block.geometry.x):
    #             if j == len(block.geometry.x):  # catch last case
    #                 x1 = block.geometry.x(j)
    #                 x2 = block.geometry.x(0)
    #                 y1 = block.geometry.y(j)
    #                 y2 = block.geometry.y(0)
    #             else:
    #                 x1 = block.geometry.x(j)
    #                 x2 = block.geometry.x(j + 1)
    #                 y1 = block.geometry.y(j)
    #                 y2 = block.geometry.y(j + 1)
    #
    #             # normal of line
    #             z = np.sqrt((y2 - y1) ^ 2 + (x2 - x1) ^ 2)
    #             xnorm = (y1 - y2) / z
    #             ynorm = (x2 - x1) / z
    #             xtest = (x1 + x2) / 2 + xnorm * 0.1  # 0.1mm away from edge
    #             ytest = (y1 + y2) / 2 + ynorm * 0.1
    #
    #             [in_pts, on_pts] = inpolygon(xtest, ytest, block.geometry.x, block.geometry.y)
    #
    #             if in_pts == 1 and on_pts == 0:
    #                 labelXY[0, :] = [xtest, ytest]
    #                 break
    #             else:
    #                 xtest = (x1 + x2) / 2 - xnorm * 0.1
    #                 ytest = (y1 + y2) / 2 - ynorm * 0.1
    #
    #                 [in_pts, on_pts] = inpolygon(xtest, ytest, block.geometry.x, block.geometry.y)
    #                 if in_pts == 1 and on_pts == 0:
    #                     labelXY[0, :] = [xtest, ytest]
    #                     break
    #
    #             if j == len(block.geometry.x):
    #                 print('Error - block label not found')
    pass


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


class Material:
    def __init__(self):
        # material for magnetics problems
        # mi_addmaterial(’matname’, mu_x, mu_y, Hc, J, Cduct, Lam_d, Phi_hmax, Lam_fill, LamType, Phi_hx, Phi_hy, nstr, dwire)
        # adds a new material with called ’matname’ with the material properties:
        # – mu_x Relative permeability in the x- or r-direction.
        # – mu_y Relative permeability in the y- or z-direction.
        # – H_c Permanent magnet coercivity in Amps/Meter.
        # – J Applied source current density in Amps/mm2. TODO: check if results are correct with Amps/m2
        # – Cduct Electrical conductivity of the material in MS/m.
        # – Lam_d Lamination thickness in millimeters.
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
        self.H_c = None
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
        self.H_c = Hc
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


class Block:
    name = None
    geometry = None
    material = None
    excitation = None
    label = None
    group = None  # A member of group number group

    # TODO: how to get private -> set not allowed
    geometry_xy = None

    def __init__(self, name):
        self.name = 'foo'
        self.geometry = {}
        self.material = Material()
        self.excitation = None  # MagneticCircuit(circuit_name=None, current=None, circuittype=None)
        self.label = Label(None, None)

    # @geometry_xy.getter
    # def geometry_xy(self):
    #     pass

    def set_label_position(self):
        # [self.label.x_coord, self.label.y_coord] = find_in_pts_in_block(block=block)
        pass

    def xmirror(self):
        self.geometry_xy[:, 2] = -1 * self.geometry_xy[:, 2]
        self.label.y_coord = -self.label.y_coord

    def move(self, x_offset, y_offset):
        self.geometry_xy = self.geometry_xy + [x_offset, y_offset]


class Ecore(Block):
    def __init__(self, name: str, A: float, B: float, C: float, D: float, E: float, F: float, ferrite: Material):
        super().__init__(name=name)
        self.geometry['A'] = A
        self.geometry['B'] = B
        self.geometry['C'] = C
        self.geometry['D'] = D
        self.geometry['E'] = E
        self.geometry['F'] = F
        self.geometry_xy = self.create_geometry_polyline()
        self.material = Ferroxcube_3C95()
        self.excitation = None  # no current in core
        self.label = Label(self.geometry['A'] / 2, self.geometry['D'] / 2)

    def create_geometry_polyline(self):
        # E-core
        xy_polyline = np.array([[0, 0],
                                [0, self.geometry['D']],
                                [(self.geometry['A'] - self.geometry['B']) / 2, self.geometry['D']],
                                [(self.geometry['A'] - self.geometry['B']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] - self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] - self.geometry['C']) / 2, self.geometry['D']],
                                [(self.geometry['A'] + self.geometry['C']) / 2, self.geometry['D']],
                                [(self.geometry['A'] + self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['B']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['B']) / 2, self.geometry['D']],
                                [self.geometry['A'], self.geometry['D']],
                                [self.geometry['A'], 0]])

        return xy_polyline

    def reduce_middle_leg_length(self, distance):
        xy_polyline = np.array([[0, 0],
                                [0, self.geometry['D']],
                                [(self.geometry['A'] - self.geometry['B']) / 2, self.geometry['D']],
                                [(self.geometry['A'] - self.geometry['B']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] - self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] - self.geometry['C']) / 2, self.geometry['D'] - distance],
                                [(self.geometry['A'] + self.geometry['C']) / 2, self.geometry['D'] - distance],
                                [(self.geometry['A'] + self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['C']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['B']) / 2,
                                 (self.geometry['D'] - self.geometry['E'])],
                                [(self.geometry['A'] + self.geometry['B']) / 2, self.geometry['D']],
                                [self.geometry['A'], self.geometry['D']],
                                [self.geometry['A'], 0]])
        # x_vec = xy_polyline[:, 0]
        # y_vec = xy_polyline[:, 1]

        return xy_polyline

    def draw_geometry_xy(self, headerFEM):
        headerFEM.create_polygon(self.geometry_xy)

    def draw_geometry_xyz(self, headerFEM):
        # TODO: take from maxwell 3D toolbox
        pass


class Icore(Block):
    def __init__(self, name: str, A: float, F: float, It: float, ferrite: Material):
        super().__init__(name=name)
        self.geometry['A'] = A
        self.geometry['F'] = F
        self.geometry['It'] = It
        self.geometry_xy = self.create_geometry_polyline()
        self.material =
        self.excitation = None  # no current in core
        self.label = Label(self.geometry['A'] / 2, self.geometry['It'] / 2)

    def create_geometry_polyline(self):
        # Icore
        xy_polyline = np.array([[0, 0],
                                [0, self.geometry['It']],
                                [self.geometry['A'], (self.geometry['It'])],
                                [self.geometry['A'], 0]])

        return xy_polyline

    def draw_geometry_xy(self, headerFEM):
        headerFEM.create_polygon(self.geometry_xy)

class Winding():
    def __init__(self):
        self.block_list = []

    def generate_conductor_blocks_list(self):
        pass

class PCBwinding(Winding):
    def __init__(self, hcu):
        super().__init__(name=name)
        self.copper_stack = hcu
        self.isolation_stack = np.array([0.3, 0.3, 0.3, 0.3, 0.3]) * 1e-3
        self.prim = Primary()
        self.sec = Secondary()
        self.block_list = self.generate_conductor_blocks_list(self)

    def generate_conductor_blocks_list(self):
        drawEwinding


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

                x1 = 0
                y1 = 0

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


class Air(Material):
    def __init__(self):
        super().__init__()
        self.name = 'air'
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

class Ferrite(Material):
    def __init__(self):
        super().__init__()
        self.name = 'ferrite'
        self.mu_x = 3000
        self.mu_y = 3000
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
    name = None
    x_coord = None
    y_coord = None

    def __init__(self, x_coord, y_coord):
        self.x_coord = y_coord
        self.y_coord = y_coord


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
    filename = None
    problem = None
    model = None

    def __init__(self):
        self.filename = "pippo"
        self.problem = FEMMMagneticProblem()

    @staticmethod
    def open():
        # The package must be initialized with the openfemm command.
        femm.openfemm()

    @staticmethod
    def close():
        femm.closefemm()

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
        # addBoundariesAir(blocks, r_boundary)
        if self.problem.solver == 'magnetic':
            # TODO for general geometry
            pass
            # # Define an "open" boundary condition using the built-in function:
            # ## mi makeABC(n,R,x,y,bc)
            # #      n = number of shells (between 1 and 10)
            # #      R = the radius of solution domain
            # #      (x,y) = center of the solution domain
            # #      bc = 0 for a Dirichlet outer edge or 1 for a Neumann outer edge.
            #
            #
            # # label air
            # femm.mi_clearselected();
            # femm.mi_getmaterial('Air')
            # femm.mi_addblocklabel(min_x - widthX * 0.1, min_y - heightY * 0.1);
            # femm.mi_selectlabel(min_x - widthX * 0.1, min_y - heightY * 0.1);
            # femm.mi_setblockprop('Air', 1, 0, 0, 0, air_id, 0) # 1 = Automesh
            # femm.mi_attachdefault()
            #
            # # set boundary
            # femm.mi_makeABC(7, r_boundary, mid_x, mid_y, 0)
            # femm.mi_clearselected()

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

    def set_block_proprities(self, block: Block):
        # Use either automesh or minmesh
        automesh = 1
        minmesh = 0.5  # minimum mesh size in mm (No effect if automesh = 1)
        if self.problem.solver == 'magnetic':
            if block.material == 'linear':
                femm.mi_selectlabel(block.label.x_coord, block.label.y_coord)
                self.add_material(material=block.material)
                femm.mi_setblockprop(block.material.name, automesh, minmesh, 0, 0, block.group, 0);

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
        # if self.problem.solver == 'magnetic':
        #     # If we were interested in the flux density at specific positions,
        #     # we could inquire at specific points directly:
        #     flux = femm.mo_getb(x, y)
        #     return flux[1]
        pass

    def get_electrical_param(self, circuit: MagneticCircuit):
        # if self.problem.solver == 'magnetic':
        #     # The program will report the terminal properties of the circuit:
        #     # current, voltage, and flux linkage
        #     [current, voltage, phi] = femm.mo_getcircuitproperties(circuit.name)
        #     return [current, voltage, phi]
        pass

    def get_inductance(self, circuit: MagneticCircuit):
        # if self.problem.solver == 'magnetic':
        #     # If we were interested in inductance, it could be obtained by
        #     # dividing flux linkage by current
        #     [current, voltage, phi] = femm.mo_getcircuitproperties(circuit.name)
        #     inductance = phi / current
        #     return inductance
        pass

    def get_flux_along_path(self, x_path: list, y_path: list):
        # if self.problem.solver == 'magnetic':
        #     # Or we could, for example, plot the results along a path
        #     flux_path = []
        #     for x, y in zip(x_path, y_path):
        #         flux = femm.mo_getb(x, y)
        #         flux_path.append(flux[1])
        #     return flux_path
        pass

    def solve(self):
        if self.problem.solver == 'magnetic':
            # Now,analyze the problem and load the solution when the analysis is finished
            femm.mi_analyze()
            femm.mi_loadsolution()

    # def draw_block_list(self):
    #     # Use either automesh or minmesh
    #     automesh = 1
    #     minmesh = 0.5  # minimum mesh size in mm(No effect if automesh = 1)
    #
    #     for item in self.blocks_list:
    #         item.geometry.draw_femm()
    #         item.set_material_femm(item.material)
    #         femm.mi_setblockprop(item.material.name, automesh, minmesh, 0, 0, item, 0)

    def draw_polyline(self, polyline: np.array):
        mi_drawpolygon(polyline)

    def add_label(self, block: Block):
        femm.mi_addblocklabel(block.label.xpos, block.label.ypos)

    def select_label(self, block: Block):
        femm.mi_selectlabel(block.label.xpos, block.label.xpos)

    def add_material(self, material: Material):
        if self.problem.solver == 'magnetic':
            try:
                femm.mi_addmaterial(material.name,
                                    material.mu_x,
                                    material.mu_y,
                                    material.H_c,
                                    material.current_density,
                                    material.sigma,
                                    material.lam_thickness,
                                    material.hyst_lag_angle,
                                    material.fill_factor_lamination,
                                    material.lamination_type,
                                    material.hyst_lag_x,
                                    material.hyst_lag_y,
                                    material.nstrand,
                                    material.dstrand)
            except AttributeError:
                pass

    def add_non_linear_material(self, material: Material):
        if self.problem.solver == 'magnetic':
            try:
                femm.mi_addmaterial(material.name,
                                    material.mu_x,
                                    material.mu_y,
                                    material.perm_magn_coerc,
                                    material.current_density,
                                    material.sigma,
                                    material.lam_thickness,
                                    material.hyst_lag_angle,
                                    material.fill_factor_lamination,
                                    material.lamination_type,
                                    material.hyst_lag_x,
                                    material.hyst_lag_y,
                                    material.nstrand,
                                    material.dstrand)
                for item in range(0, len(material.bdata)):
                    femm.mi_addbhpoint(material.name, material.bdata[item], material.hdata[item])
            except AttributeError:
                pass

    def draw_block(self, block):
        self.add_material(material=block.material)
        self.draw_polyline(polyline=block.geometry_xy)
        self.add_block_label(block.label.x_coord, block.label.y_coord)
        self.set_block_proprities(block=block)

    # def createEcore(core, material_name, gap):
    #     c = core
    #     mi_drawpolygon(np.array([[0, 0], [0, c.D], [(c.A - c.B) / 2, c.D], [(c.A - c.B) / 2, c.D - c.E],
    #                              [(c.A - c.C) / 2, c.D - c.E], [(c.A - c.C) / 2, c.D], [(c.A + c.C) / 2, c.D],
    #                              [(c.A + c.C) / 2, c.D - c.E], [(c.A + c.C) / 2, c.D - c.E],
    #                              [(c.A + c.B) / 2, c.D - c.E], [(c.A + c.B) / 2, c.D],
    #                              [c.A, c.D], [c.A, 0]]))
    #     femm.mi_addblocklabel(c.A / 2, c.D / 2)
    #     femm.mi_selectlabel(c.A / 2, c.D / 2)
    #     femm.mi_setblockprop(material_name, 0, 0.5, 0, 0, 0, 0)

    # def drawRectWithCenterWidthHeight(xc, yc, w, h, material: Material):
    #     pts = np.array(
    #         [[xc - w / 2, yc - h / 2], [xc - w / 2, yc + h / 2], [xc + w / 2, yc + h / 2], [xc + w / 2, yc - h / 2]])
    #     mi_drawpolygon(pts)
    #     femm.mi_addblocklabel(xc, yc)
    #     femm.mi_selectlabel(xc, yc)
    #     femm.mi_setblockprop(material.name, 0, 0.5, 0, 0, 0, 0)


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


def drawEwindingHalfTurn(Core, PCB, winding, material_name):
    # This funtion draws the half turn planar windings in FEMM
    pass

def createEEcoreWith3Gaps(core: Ecore, gap):
    EblockBot = copy.deepcopy(core)
    EblockTop = copy.deepcopy(core)
    EblockTop.xmirror()
    EblockTop.move(x_offset=0, y_offset=core.geometry['D'] + gap)
    return EblockBot, EblockTop


def createEIcoreWith3Gaps(ecore: Ecore, plt_core: Icore, gap):
    Eblock = copy.deepcopy(ecore)
    Iblock = copy.deepcopy(plt_core)
    Iblock.move(x_offset=0, y_offset=Eblock.geometry['D'] + gap)
    return Eblock, Iblock


def drawEEcoreGapMidLeg(core: Ecore, gap):
    EblockBot = copy.deepcopy(core)
    EblockTop = copy.deepcopy(core)
    EblockBot.reduce_middle_leg_length(gap / 2)
    EblockTop.reduce_middle_leg_length(gap / 2)
    EblockTop.xmirror()
    EblockTop.move(x_offset=0, y_offset=core.geometry['D'])
    return EblockBot, EblockTop


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


# def test_short_CT(Core, winding, id, f, PCB, g):
#     femm.mi_getmaterial('Air')
#
#     if g == 0:
#         femm.mi_addblocklabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
#         femm.mi_selectlabel((Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
#         femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)
#
#         femm.mi_addblocklabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
#         femm.mi_selectlabel(Core.A / 2 + (Core.B - Core.C) / 2, PCB.d_core / 2 + Core.D - Core.E)
#         femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)
#
#     femm.mi_addblocklabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
#     femm.mi_selectlabel(Core.A * np.cos(np.pi / 3), Core.A * np.sin(np.pi / 3))
#     femm.mi_setblockprop('Air', 1, 0, 0, 0, 99, 0)
#
#     femm.mi_probdef(f, 'meters', 'planar', 1e-8, Core.F)
#     femm.mi_makeABC(7, 2 * Core.A, Core.A / 2, Core.D / 2, 0)
#
#     femm.mi_saveas(''.join([os.getcwd(), '\\Output\\FEM\\test_short.fem']))
#     femm.mi_createmesh()
#
#     n = []
#     Rac = []
#
#     for i in range(len(winding)):
#         n[i] = np.sum(winding[id - 1].n_turn_layer) / winding[id - 1].parallel / np.sum(winding[i].n_turn_layer) * \
#                winding[i].parallel
#         if i == id:
#             if winding[i].half:
#                 set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
#             else:
#                 set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 1 / winding[i].parallel)
#         else:
#             if winding[i].half:
#                 set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer,
#                                  -n[i] / winding[i].parallel)
#             else:
#                 set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, -n[i] / winding[i].parallel)
#
#     femm.mi_analyze(1)
#     femm.femm.mi_loadsolution()
#     for i in range(len(winding)):
#         if i == id:
#             femm.mo_groupselectblock(i)
#             Rac[i] = 2 * np.real(femm.mo_blockintegral(4))
#             femm.mo_clearblock()
#         else:
#             femm.mo_groupselectblock(i)
#             femm.mo_groupselectblock(i + 1)
#             Rac[i] = 2 * np.real(femm.mo_blockintegral(4)) / n[i] ^ 2
#             femm.mo_clearblock()
#
#     femm.mo_groupselectblock(0)
#     femm.mo_groupselectblock(99)
#     L_leak = 4 * np.real(femm.mo_blockintegral(2))
#     femm.mo_clearblock()
#
#     femm.mo_resize(800, 800)
#     femm.mo_zoomnatural()
#     femm.mo_zoomin()
#     femm.mo_savebitmap(''.join([os.getcwd(), '\\img\\test_short.bmp']))
#     femm.mo_savemetafile(''.join([os.getcwd(), '\\img\\test_short.eps']))
#     # reset currents to 0
#     for i in range(len(winding)):
#         if winding[i].half:
#             set_circuit_half(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
#         else:
#             set_circuit(winding[i].name, winding[i].layers, winding[i].n_turn_layer, 0)
#     return [Rac, L_leak]

class Transformer:
    def __init__(self):
        ## electrical paramaters
        self.primary_number_of_turns = 3  # Primary number of turns
        self.secondary_number_of_turns = 3
        self.frequency = 200e3

        ## winding stack description
        self.n_layer = 6
        self.tlp = np.array([1, 1, 1, 0, 0, 0])  # description of turn of primary in layer stack ex 2T in 2nd layer
        self.np_HV = 1  # number of turn in parallel for the primary
        self.tls = np.array([0, 0, 0, 1, 1, 1])
        self.np_LV = 1  # number of turn in parallel for the sec

        d_core2pri = 1e-3  # distance core to primary
        d_core_s = 1e-3  # distance core to secondary
        cr_winding = 0.3e-3  # creepage between 2 traces

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

        # winding resistance
        tw_p = (ww - cr_winding * (np.ceil(tlp) - 1) - 2 * d_core2pri) / tlp  # trace width in each primary layer
        # the same for secondary winding resistance
        w_mean_s = (ww - cr_winding * (np.ceil(tls) - 1) - 2 * d_core_s) / tls
        # prim winding resistance
        Rwp = np.sum(
            tlp * lw / tw_p / myPCBwinding.copper_stack / sigmaCu_20) / np_HV ** 2  # np.sum each turn resistance
        Rws = np.sum(tls * lw / w_mean_s / myPCBwinding.copper_stack / sigmaCu_20) / np_LV ** 2
        # the sec resistance reflected to primary
        Rws2p = n_HV ** 2 / np_LV ** 2 * Rws
        Rt = Rwp + Rws2p


def test_trafo_E22_6_16():
    ## constant definition

    ## FEM init
    # open a new femm document
    trafo_FEM = FEMM()
    trafo_FEM.open()
    trafo_FEM.new_magnetic_planar_problem()
    trafo_FEM.set_frequency(frequency=0.0)

    ## core description
    # core dimensions as defined in PlanarFEMM_diagram.png
    # for E22/6/16 in E/PLT arrangement
    # https://www.ferroxcube.com/en-global/download/download/11
    # Dimension defined as in PlanarFEMM_diagram.png (NOTE THAT planarFEMM TAKES DIMENSIONS IN MM)
    E22616_material = Ferroxcube_3C95()
    E22616 = Ecore(name='E22616', A=21.8e-3, B=16.8e-3, C=5e-3, D=5.7e-3, E=3.2e-3, F=15.8e-3, ferrite=E22616_material)
    PLT22616 = Icore(name='PLT22616', A=21.8e-3, F=15.8e-3, It=2.5e-3, ferrite=E22616_material)
    myPCBwinding = PCBwinding(hcu=np.array([105e-6, 105e-6, 105e-6, 105e-6, 105e-6, 105e-6]))  # copper layer thickness

    # myPrimary = Primary()
    # mySecondary = Secondary()

    femm.mi_makeABC(7, 2 * E22616.geometry['A'], E22616.geometry['A'] / 2, E22616.geometry['D'] / 2, 0)

    [fem_E22616, fem_PLT22616] = createEIcoreWith3Gaps(ecore=E22616, plt_core=PLT22616, gap=0)
    trafo_FEM.draw_block(block=fem_E22616)
    trafo_FEM.draw_block(block=fem_PLT22616)

    # drawEIcoreGapEachLeg(core=E22616, t=PLT22616.t, material=E22616_material, gap=gap)
    # drawEIcoreGapMidLeg(E22616, PLT22616.t, E22616_material, gap)
    # drawEEcoreGapEachLeg(core=E22616,  t=PLT22616.t, material=E22616_material, gap=gap)
    # drawEEcoreGapMidLeg(E22616, E22616, E22616_material, gap)

    # drawEwinding(E22616, myPCBwinding, myPrimary, 'Copper')
    # drawEwinding(E22616, myPCBwinding, mySecondary, 'Copper')
    # time.sleep(2)
    #
    # Rdc_p, L_self_p = test_open(E22616, [myPrimary, mySecondary], myPrimary.id, myPCBwinding, gap)
    # Rdc_s, L_self_s = test_open(E22616, [myPrimary, mySecondary], myPrimary.id, myPCBwinding, gap)
    #
    # print('Rdc_p= %e' % Rdc_p)
    # print('Rdc_s= %e' % Rdc_s)
    #
    # print('L_self_p= %e' % L_self_p)
    # print('L_self_s= %e' % L_self_p)
    #
    # Rac, L_leak = test_short(E22616, [myPrimary, mySecondary], myPrimary.id, fs, myPCBwinding, gap)
    # ratioRacRdc = (Rac[0] + Rac[1] * n_HV ** 2) / (Rdc_p + n_HV ** 2 * Rdc_s)
    # L_leak_mm = L_leak / E22616.F
    #
    # L_leak = L_leak_mm * lw
    # k = np.sqrt(1 - L_leak / L_self_p)

    # test_saturation(E22616, [myPrimary, mySecondary], myPrimary.id, myPCBwinding, Im/np.sqrt(2), gap)
    pass


if __name__ == "__main__":
    test_trafo_E22_6_16()
