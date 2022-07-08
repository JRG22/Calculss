import csv

import numpy as np
from radioStation import *
import fonctions_generales as fg


# TODO mettre les commentaires en anglais

#  Caractéristiques et fonctions relatifs à la station de contrôle au sol
import UAS

def get_gain_from_angle(angle):
    """Retourne la valeur de gain correspondant à l'angle

    Parameters
    ----------
    angle : float
        Numéro du canal d'émission
    Returns
    -------
    float
        gain
    """
    gain = np.ones(1) * 3
    gain = np.where(angle < 75, 4, gain)
    gain = np.where(angle < 64, 9, gain)
    gain = np.where(angle < 32, 14, gain)
    gain = np.where(angle < 16, 16.5, gain)
    gain = np.where(angle < 11.5, 19.5, gain)
    gain = np.where(angle < 7, 22, gain)
    gain = np.where(angle < 3.5, 22.5, gain)
    gain = np.where(angle < 2.5, 21, gain)
    gain = np.where(angle < 1.5, 21.5, gain)
    gain = np.where(angle < 0.5, 21.5, gain)
    return gain


class BS(radioStation):
    """Définition de la station de base"""

    def __init__(self, lat, long, alt, nb_canal=1):
        # TODO put option bandwidth
        super().__init__(lat, long, alt, f=5030e6 + 250e3 * nb_canal)

        self.criterion = None
        self.ge = 3
        self.fl = 3
        self.figure_noise = 7
        self.canal = nb_canal
        self.N = -113

    def set_gain_from_angle(self, angle):
        """
        Fonction calculant le gain de la ground station a partir des caractéristiques définies [Tableau d'entrées non testé]
        :param angle: Angle en degrés entre la droite horizon et la droite ground station - drone
        :type angle: float (degrés)
        :return: Gain de l'antenne
        :rtype: int (dB)
        """
        self.gain = get_gain_from_angle(angle)
        return get_gain_from_angle(angle)

    def calcul_i(self, drone, tab_drones_i):

        self.interf = np.ones(int(np.size(drone.trajectoire_obj.temps))) * self.N

        for drone_i in tab_drones_i:
            tab_interf = fg.calcul_pr_radiostations(drone_i, self)
            for i in range(int(min(np.size(drone.trajectoire_obj.temps), np.size(tab_interf)))):
                self.interf[i] = fg.sum_dbm(np.array([self.interf[i], tab_interf[i]]))

    def calcul_c(self, drone):

        self.C = fg.calcul_pr_radiostations(drone, self)

    def export_char_ground_station(self):

        with open('results_ground_station.csv', mode='w', newline='', ) as ground_station:
            results_ground_station = csv.writer(ground_station, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_ground_station.writerow(
                ['Puissance reçue par le satellite', 'interférence reçue par le satellite', 'C / N+I'])
            i = 0
            while i < np.size(self.C):
                results_ground_station.writerow([self.C[i], self.interf[i], self.criteria_c_i])
                i += 1

    def calcul_criterion(self):
        self.criterion = self.C - self.interf

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
        """Calcul de la puissance non voulue émise par les drones interférents reçue par la ground statio

                       Parameters
                       ----------

                       Returns
                       -------
                       float
                           Puissance reçue par la ground station émise par les drones interférents
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
                coord_emission = drone_i.trajectoire_obj.coordonnees[:, i]
                distance = fg.calcul_distance_lla(coord_emission, coord_reception)
                lp = fg.fsl(self.f, distance)
                if is_first_drone:
                    is_first_drone = False

                    self.interf[i] = fg.p_received(pe, ge, gr, lp, fle, flr)
                else:

                    self.interf[i] = fg.sum_dbm(np.array([self.interf[i], fg.p_received(pe, ge, gr, lp, fle, flr)]))

        return self.interf
