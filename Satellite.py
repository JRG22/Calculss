import csv
import random as rd
from ANFRpy.geometry.ENU_to_ECEF import *
from ANFRpy.geometry.ECEF_to_WGS84 import *

import fonctions_generales as fg
from radioStation import *

#  Caractéristiques et fonctions relatives au drone (UAS)

'''=============================================================================
 |
 |       Author:  Julien Roussel-Galle
 |       Date: 
 +-----------------------------------------------------------------------------
 |
 |  Description:  Class Satellite with general characteristics
 |
 |       
 |
 *===========================================================================*/
'''


# TODO coder cette partie du satellite

class Satellite(radioStation):
    """Définition d'un satellite"""

    def __init__(self, lat, long, alt=800000, pe=40):
        super().__init__(lat, long, alt)
        self.GE = 37.8
        self.FL = 1
        self.figure_noise = 7
        self.latitude, self.longitude, self.altitude = lat, long, alt  #
        self.N = -113


    def set_interf(self, interference):

        for i in range(int(np.size(interference))):
            self.interf[i] = fg.sum_dbm(np.array([self.interf[i], interference[i]]))

    def set_pr(self, pr):
        self.interf = np.ones(int(np.size(pr))) * self
        for i in range(int(np.size(pr))):
            self.C[i] = fg.sum_dbm(np.array([self.C[i], pr[i]]))

    def calcul_i(self, drone, tab_drones_i):

        self.interf = np.ones(int(np.size(drone.trajectoire_obj.temps))) * self.N

        for drone_i in tab_drones_i:
            tab_interf = fg.calcul_pr_radiostations(drone_i, self)
            for i in range(int(min(np.size(drone.trajectoire_obj.temps), np.size(tab_interf)))):
                self.interf[i] = fg.sum_dbm(np.array([self.interf[i], tab_interf[i]]))

    def calcul_c(self, drone):

        self.C = fg.calcul_pr_radiostations(drone, self)

    def export_char_satellite(self):
        with open('results_satellite.csv', mode='w', newline='', ) as results_satellite:
            results_satellite = csv.writer(results_satellite, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_satellite.writerow(
                ['Puissance reçue par le satellite', 'interférence reçue par le satellite', 'C / N+I'])
            i = 0
            while i < np.size(self.C):
                results_satellite.writerow([self.C[i], self.interf[i], self.criteria_c_i[i]])
                i += 1

    def get_probability_message_lost(self, limite, nb_drones, drone):
        number_points_interf = 0
        self.criteria = np.ones(np.size(self.C))
        for i in range(np.size(self.C)):

            self.criteria[i] = self.interf[i] - self.C[i]
            if self.criteria[i] < limite:
                number_points_interf += 1

        p_single_collision = drone.t_downlink * 2 / drone.t_period
        p_multiple_collision = 1 - (1 - p_single_collision) ** nb_drones
        self.probability_message_lost = (number_points_interf / np.size(
            self.criteria)) * p_multiple_collision
        print("'UAS : critère à respecter C/N+I", limite)
        print("UAS : proba de dépassement du critère :",
              round((number_points_interf / np.size(self.criteria)), 2),
              " \t proba d'émission en même temps(nombre de drones =", nb_drones, ") :, ",
              1 - (1 - p_single_collision) ** nb_drones)

        return self.probability_message_lost

    def calcul_interf(self, tab_drone_i):
        """Calcul de la puissance non voulue émise par les drones interférents reçue par le satellite

                       Parameters
                       ----------

                       Returns
                       -------
                       float
                           Puissance reçue par le satellite émise par les drones interférents
                   """


        p_uas_v, gr, flr, canal_v = self.get_parameters()
        coord_reception = self.get_coordinates()

        tmax = np.size(tab_drone_i[0].trajectoire_obj.temps)
        for drone_i in tab_drone_i:
            if tmax < np.size(drone_i.trajectoire_obj.temps):
                tmax = np.size(drone_i.trajectoire_obj.temps)

        self.interf = np.ones(tmax)
        is_first_drone = True
        for drone_i in tab_drone_i:
            p_uas_i, ge, fle, canal_v = drone_i.get_parameters()
            pe = p_uas_i
            for i in range(np.size(drone_i.trajectoire_obj.temps)):
                coord_emission = drone_i.trajectoire_obj.coordonnees[:,i]
                distance = fg.calcul_distance_lla(coord_emission, coord_reception)
                lp = fg.fsl(self.f, distance)
                if is_first_drone :
                    is_first_drone = False

                    self.interf[i] = fg.p_received(pe, ge, gr, lp, fle, flr)
                else :

                    self.interf[i] = fg.sum_dbm(np.array([self.interf[i],fg.p_received(pe, ge, gr, lp, fle, flr)]))


    def calcul_C(self, drone):

        p_uas, ge, fle, canal = self.get_parameters()
        p_bs, gr, flr, canal_i = drone.get_parameters()
        t_max = np.size(drone.trajectoire_obj.temps)
        self.C = np.ones(t_max)
        distance = fg.calcul_distance_lla(self.get_coordinates(), drone.trajectoire_obj.coordonnees)

        lp = fg.fsl(self.f, distance)
        pe = p_uas

        ge = 20
        pr = fg.p_received(pe, ge, gr, lp, fle, flr)
        self.C = pr
