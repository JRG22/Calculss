import scipy.constants as scc
from ANFRpy.geometry import Calcul_ECEF_ENU

from ANFRpy.geometry.WGS84_to_ECEF import *

import UAS as UAS
import GroundStation as Bs
import radioStation as rS

import matplotlib.pyplot as plt


class trajectoire:
    def __init__(self, laps_temps, coordonnees):
        self.coordonnees = coordonnees
        self.laps_temps = laps_temps
        self.temps = np.array([])
        self.set_temps()
        self.service_range = None
        self.distance_ground_station = None
        self.distance_satellite = None
        self.angle_ground_station = None
        self.angle_satellite = None
        self.pr_drone = None
        self.pr_ground_station = None
        self.pr_satellite = None
        self.distance_Interferer = np.zeros(np.size(self.temps))
        self.N = -113
        self.NplusI = np.ones(np.size(self.temps)) * self.N
        self.criteria = None

    def set_temps(self):
        self.temps = np.ones(int(np.size(self.coordonnees) / 3))
        for i in range(int(np.size(self.coordonnees) / 3)):
            self.temps[i] = i * self.laps_temps

    def plot_pr_fct_temps(self):
        plt.plot(self.temps, self.pr_drone)

    def plot_c_surn_fct_temps(self):
        plt.plot(self.temps, self.pr_drone - self.N)


