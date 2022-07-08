#  Caractéristiques et fonctions relatives au drone (UAS)
from ANFRpy.geometry.ENU_to_ECEF import *

import fonctions_generales as fg
import matplotlib.pyplot as plt
from trajectoire import trajectoire
from radioStation import radioStation

# import radioStation as rS
plt.rcParams['axes.grid'] = True


def masques_emission(px, pas, offset=0, affichage=False):
    """

    :param px: Puissance transmise en dBm
    :param pas: pas d'échantillonage du masque en hz
    :param offset: Décalage de canaux entre le masque d'émission et de réception
    :param affichage: True pour plot le masque
    :return: array contenant les valeurs du masque
    """
    if offset < 0 and offset > 244:
        print("ERROR")
    channel_width = 250e3  # Bandwidth in kHZ
    offset_max = (channel_width / 2) * 31  # Nombre de *125 doit être impair
    offset_min = -offset_max
    if offset != 0:
        offset_max += channel_width * offset

    f = np.arange(offset_min, offset_max, pas)
    y_wk_points = np.array([-126, -126, -120, -104, -84, -54, -54, -84, -104, -120, -126, -126])
    x_wk_points = np.array(
        [offset_min, -2000e3, -500e3, -375e3, -125e3, -125e3, 125e3, 125e3, 375e3, 500e3, 2000e3, offset_max])
    SEM = np.interp(f, x_wk_points, y_wk_points)
    if affichage:
        plt.plot(f, SEM)
        plt.title("Emission Mask")
        plt.xlabel("frequency in Hz")
        plt.ylabel("Filtred Power / Hz in dB")
        pas = channel_width
        if offset_max > 5e6:
            pas = 5e6
        a = np.arange(-offset_max, -0.5 * channel_width, pas)
        b = np.array([-pas / 2, pas / 2])
        c = np.arange(1.5 * channel_width, offset_max, pas)
        xtick = np.concatenate([a, b, c])
        plt.xticks(xtick)
        plt.show()
    return px + SEM + 10 * np.log10(pas)


def masques_reception(pas, offset=0, affichage=False):
    channel_width = 250e3
    offset_max = (channel_width / 2) * 31  # doit etre impair
    offset_min = -offset_max
    if offset != 0:
        offset_min -= offset * channel_width
    mask = np.arange(offset_min, offset_max, pas)
    f = np.arange(offset_min, offset_max, pas)
    mask[abs(f) >= 0] = 0
    mask[abs(f) > 125e3] = -23
    mask[abs(f) > 375e3] = -43
    mask[abs(f) > 625e3] = -57
    mask[abs(f) > 2000e3] = -63

    if affichage:

        plt.plot(f, mask)
        plt.title("Reception Mask")
        plt.xlabel("frequency  in Hz")
        plt.ylabel("dB")
        pas = channel_width
        if offset_max > 5e6:
            pas = 5e6
        a = np.arange(-offset_max, -0.5 * channel_width, pas)
        b = np.array([-pas / 2, pas / 2])
        c = np.arange(1.5 * channel_width, offset_max, pas)
        xtick = np.concatenate([a, b, c])
        plt.xticks(xtick)
        plt.show()

    return mask


class UAS(radioStation):
    """Definition d'un drone"""

    # TODO : Remplacer l'ancien tableau de trajectoires par le nouveau

    def __init__(self, lat, long, alt, pe=40):
        super().__init__(lat, long, alt, pe)
        self.B = 20e6  # Peut être modifié
        self.ge = 3
        self.fl = 2
        self.noise_figure = 7
        self.trajectoire = np.array([])  # UAS trajectory : Coordinates array
        self.speed_m_s = 28  # UAS Speed in m/s
        self.trajectoire_ps = None
        self.trajectoire_obj = None
        self.pourcentage_reussite = 0
        self.t_uplink = 0.1  # en micro secondes
        self.t_downlink = 0.5
        self.t_period = 50
        self.duty_cycle_uplink = 0.2 / 50
        self.duty_cycle_downlink = 4 / 50
        self.C_terrestrial = None
        self.C_satellite = None
        self.c_i = None
        self.i_n = None

    def set_trajectoire_bs(self, coord_bs, angle_entry):
        """Définit la trajectoire du drone au dessus de la bs, dans la ligne de visibilité
        de la bs

        Parameters
        ----------
        coord_bs: float[3]
            Coordonnées de la station de base
        angle_entry: float
            Angle d'entrée du drone dans zone de la station de base
        Returns
        -------


        """
        laps_temps = 60
        bs_height_m = coord_bs[2]
        r_m = 6371008.7714  # Rayon moyen de la Terre en mètres
        a_m = 6378137  # demi grand axe en mètre
        # Angles nécessaires pour la suite du calcul
        alpha1 = np.degrees(np.arccos(r_m / (r_m + bs_height_m)))
        alpha2 = np.degrees(np.arccos(r_m / (r_m + self.alt)))
        angle_entry_uas_deg = angle_entry
        difference_angles = np.random.uniform(1, 359)
        angle_exit_uas_deg = angle_entry_uas_deg + difference_angles
        angle1 = np.sin(np.radians(alpha1 + alpha2))
        angle2 = np.cos(np.radians(angle_entry_uas_deg))
        angle3 = np.sin(np.radians(angle_entry_uas_deg))
        angle4 = np.cos(np.radians(alpha1 + alpha2))
        angle5 = np.cos(np.radians(angle_exit_uas_deg))
        angle6 = np.sin(np.radians(angle_exit_uas_deg))

        # Coordonnées d'entrées et de sorties de l'avion dans la zone : ENU
        uas_coord_enu_entry = np.array([(r_m + self.alt) * angle1 * angle2,
                                        (r_m + self.alt) * angle1 * angle3,
                                        -r_m + (r_m + self.alt) * angle4])

        uas_coord_enu_exit = np.array([(r_m + self.alt) * angle1 * angle5,
                                       (r_m + self.alt) * angle1 * angle6,
                                       -r_m + (r_m + self.alt) * angle4])

        # Coordonnées d'entrées et de sorties de l'avion dans la zone WGS84
        uav_coord_ecef_m_entry = Calcul_ENU_ECEF(uas_coord_enu_entry[0], uas_coord_enu_entry[1], uas_coord_enu_entry[2],
                                                 coord_bs[0], coord_bs[1], coord_bs[2])
        uav_coord_wgs84_deg_entry = Calcul_ECEF_WGS84(uav_coord_ecef_m_entry[0], uav_coord_ecef_m_entry[1],
                                                      uav_coord_ecef_m_entry[2])
        uav_coord_ecef_m_exit = Calcul_ENU_ECEF(uas_coord_enu_exit[0], uas_coord_enu_exit[1], uas_coord_enu_exit[2],
                                                coord_bs[0], coord_bs[1], coord_bs[2])
        uav_coord_wgs84_deg_exit = Calcul_ECEF_WGS84(uav_coord_ecef_m_exit[0], uav_coord_ecef_m_exit[1],
                                                     uav_coord_ecef_m_exit[2])

        distance_grand_cercle_m_entry_exit = fg.calcul_distance_grand_cercle(np.radians(uav_coord_wgs84_deg_entry),
                                                                             np.radians(uav_coord_wgs84_deg_exit))

        distance_m_entry_exit = fg.calcul_distance_lla(uav_coord_wgs84_deg_entry, uav_coord_wgs84_deg_exit)

        nb_points_aircraft_path_step_s = int(np.floor(distance_m_entry_exit / self.speed_m_s))
        nb_points_aircraft_path_step_s = int(np.floor(nb_points_aircraft_path_step_s / 60))
        is_print = False
        if is_print:
            # print("Distance grad cercle de la trajectoire ", distance_grand_cercle_m_entry_exit )
            print("On ne prends pas la distance grand cercle : Distance  de la trajectoire ",
                  distance_m_entry_exit)
            print("Vitesse du drone ", self.speed_m_s * 3.6)
            print("nb de points calculés (1 point = 1 minute) : ", nb_points_aircraft_path_step_s)
            print("temps du vol en minutes ", nb_points_aircraft_path_step_s)

            # print("j'ai fixé le nombre de points à 15")

        uav_wgs84_deg_path = np.array([
            np.linspace(uav_coord_wgs84_deg_entry[0], uav_coord_wgs84_deg_exit[0],
                        nb_points_aircraft_path_step_s),

            np.linspace(uav_coord_wgs84_deg_entry[1], uav_coord_wgs84_deg_exit[1],
                        nb_points_aircraft_path_step_s),

            np.linspace(uav_coord_wgs84_deg_entry[2], uav_coord_wgs84_deg_exit[2],
                        nb_points_aircraft_path_step_s)])
        self.trajectoire_obj = trajectoire(1, uav_wgs84_deg_path)
        # TODO : Comparer les deux lignes en dessous pour vérifier qu'elles donnent la même valeur
        coord_traj_sol_rad = np.array(
            [np.radians(self.trajectoire_obj.coordonnees[0]), np.radians(self.trajectoire_obj.coordonnees[1]), 30])
        coord_traj_sol_deg = np.array(
            [self.trajectoire_obj.coordonnees[0], self.trajectoire_obj.coordonnees[1], 30])
        # coord_drone_sol_rad = np.array([np.radians(self.trajectoire[0, :]), np.radians(self.trajectoire[1, :]), 30],dtype=object)
        # self.trajectoire_obj.service_range = fg.calcul_distance_grand_cercle(coord_traj_sol_rad, np.radians(coord_bs))
        self.trajectoire_obj.service_range = fg.calcul_distance_lla(coord_traj_sol_deg, coord_bs)
        self.trajectoire_obj.distance_ground_station = fg.calcul_distance_lla(self.trajectoire_obj.coordonnees,
                                                                              coord_bs)

        lla_drone = self.trajectoire_obj.coordonnees
        lla_base_station = coord_bs
        self.trajectoire_obj.angle_ground_station = np.vectorize(fg.calcul_angle_emetteur_lla)(lla_drone[0],
                                                                                               lla_drone[1],
                                                                                               lla_drone[2],
                                                                                               lla_base_station[0],
                                                                                               lla_base_station[1],
                                                                                               lla_base_station[2])

        self.trajectoire = uav_wgs84_deg_path

    def set_rd_coord_area_gso(self, coord_satellite):
        sat_height_m = coord_satellite[2]
        r_m = 6371008.7714  # Rayon moyen de la Terre en mètres
        a_m = 6378137  # demi grand axe en mètre
        # Angles nécessaires pour la suite du calcul
        alpha1 = np.degrees(np.arccos(r_m / (r_m + sat_height_m)))
        alpha2 = np.degrees(np.arccos(r_m / (r_m + self.alt)))

        angle_entry_uas_deg = 0  # = angle_entry

        difference_angles = np.random.uniform(1, 359)
        angle_exit_uas_deg = angle_entry_uas_deg + difference_angles
        angle1 = np.sin(np.radians(alpha1 + alpha2))
        angle2 = np.cos(np.radians(angle_entry_uas_deg))
        angle3 = np.sin(np.radians(angle_entry_uas_deg))
        angle4 = np.cos(np.radians(alpha1 + alpha2))
        angle5 = np.cos(np.radians(angle_exit_uas_deg))
        angle6 = np.sin(np.radians(angle_exit_uas_deg))

        # Coordonnées d'entrées et de sorties de l'avion dans la zone : ENU
        uas_coord_enu_entry = np.array([(r_m + self.alt) * angle1 * angle2,
                                        (r_m + self.alt) * angle1 * angle3,
                                        -r_m + (r_m + self.alt) * angle4])

        uas_coord_enu_exit = np.array([(r_m + self.alt) * angle1 * angle5,
                                       (r_m + self.alt) * angle1 * angle6,
                                       -r_m + (r_m + self.alt) * angle4])

        # Coordonnées d'entrées et de sorties de l'avion dans la zone WGS84
        uav_coord_ecef_m_entry = Calcul_ENU_ECEF(uas_coord_enu_entry[0], uas_coord_enu_entry[1], uas_coord_enu_entry[2],
                                                 coord_satellite[0], coord_satellite[1], coord_satellite[2])
        uav_coord_wgs84_deg_entry = Calcul_ECEF_WGS84(uav_coord_ecef_m_entry[0], uav_coord_ecef_m_entry[1],
                                                      uav_coord_ecef_m_entry[2])
        uav_coord_ecef_m_exit = Calcul_ENU_ECEF(uas_coord_enu_exit[0], uas_coord_enu_exit[1], uas_coord_enu_exit[2],
                                                coord_satellite[0], coord_satellite[1], coord_satellite[2])
        uav_coord_wgs84_deg_exit = Calcul_ECEF_WGS84(uav_coord_ecef_m_exit[0], uav_coord_ecef_m_exit[1],
                                                     uav_coord_ecef_m_exit[2])
        is_print = False
        if is_print:
            print("coordonnée limite a 0 degrés", uav_coord_wgs84_deg_entry)
            print("Distance de séparation", fg.calcul_distance_lla(uav_coord_wgs84_deg_entry, coord_satellite))

    def get_coordinates(self, t=0):
        if t >= np.size(self.trajectoire_obj.temps): return None
        return self.trajectoire_obj.coordonnees[:, t]

    def get_probability_message_lost(self, limite, nb_drones):
        self.trajectoire_obj.criteria = np.ones(np.size(self.trajectoire_obj.pr_drone))
        number_points_interf = 0
        for i in range(np.size(self.trajectoire_obj.temps)):
            self.trajectoire_obj.criteria[i] = self.trajectoire_obj.pr_drone[i] - self.trajectoire_obj.NplusI[i]
            if self.trajectoire_obj.criteria[i] < limite:
                number_points_interf += 1
        p_single_collision = self.t_uplink * 2 / self.t_period
        p_multiple_collision = 1 - (1 - p_single_collision) ** nb_drones
        self.probability_message_lost = (number_points_interf / np.size(
            self.trajectoire_obj.criteria)) * p_multiple_collision
        print("'UAS : critère à respecter C/N+I", limite)
        print("UAS : proba de dépassement du critère :",
              round((number_points_interf / np.size(self.trajectoire_obj.criteria)), 2),
              " \t proba d'émission en même temps(nombre de drones =", nb_drones, ") :, ",
              1 - (1 - p_single_collision) ** nb_drones)

        return self.probability_message_lost

    def calcul_interf(self, tab_drone_i):
        """Calcul de la puissance non voulue émise par les drones interférents reçue par le drone victime

                Parameters
                ----------

                Returns
                -------
                float
                    Puissance reçue non voulue par le drone
            """


        p_uas_v, gr, flr, canal_v = self.get_parameters()

        pr = np.zeros(np.size(self.trajectoire_obj.temps))
        coord_reception = self.get_coordinates()
        first_drone_interf = True
        self.interf = np.ones(np.size(self.trajectoire_obj.temps))
        for drone_i in tab_drone_i:
            p_uas_i, ge, fle, canal_v = drone_i.get_parameters()
            if first_drone_interf == True:
                tmax = min(np.size(self.trajectoire_obj.temps), np.size(drone_i.trajectoire_obj.temps))
                for i in range(tmax):
                    coord_emission = np.vectorize(UAS.get_coordinates)(drone_i, i)
                    distance = fg.calcul_distance_lla(coord_emission, coord_reception)
                    self.trajectoire_obj.distance_Interferer[i] = distance
                    lp = fg.fsl(self.f, distance)
                    pe = p_uas_i
                    pr[i] = fg.p_received(pe, ge, gr, lp, fle, flr)
                    self.interf[i] = pr[i]

            else:
                tmax = min(np.size(self.trajectoire_obj.temps), np.size(drone_i.trajectoire_obj.temps))
                for i in range(tmax):
                    coord_emission = np.vectorize(UAS.get_coordinates)(drone_i, i)
                    distance = fg.calcul_distance_lla(coord_emission, coord_reception)
                    self.trajectoire_obj.distance_Interferer[i] = distance
                    lp = fg.fsl(self.f, distance)
                    pe = p_uas_i
                    pr[i] = fg.p_received(pe, ge, gr, lp, fle, flr)
                    self.interf[i] = fg.fonctions_generales.sum_dbm(
                        np.array([self.interf[i], pr[i]]))

        return pr



    def calcul_C_terrestrial(self, ground_station):

        p_uas, ge, fle, canal = self.get_parameters()
        p_bs, gr, flr, canal_i = ground_station.get_parameters()

        distance = fg.calcul_distance_lla(self.trajectoire, ground_station.get_coordinates())
        lp = fg.fsl(self.f, distance)
        pe = p_uas

        angle_emission = self.trajectoire_obj.angle_ground_station

        """angle_emission = np.vectorize(calcul_angle_emetteur_lla)(lla_drone[0], lla_drone[1], lla_drone[2], lla_base_station[0],
                                                                 lla_base_station[1], lla_base_station[2])"""

        ge = ground_station.set_gain_from_angle(angle_emission)
        pr = fg.p_received(pe, ge, gr, lp, fle, flr)
        self.trajectoire_obj.pr_drone = pr

        self.C = pr
        self.C_terrestrial = pr

    def calcul_C_satellite(self, satellite):
        p_uas, ge, fle, canal = self.get_parameters()
        p_bs, gr, flr, canal_i = satellite.get_parameters()

        distance = fg.calcul_distance_lla(self.trajectoire, satellite.get_coordinates())

        lp = fg.fsl(self.f, distance)
        pe = p_uas

        ge = 20
        pr = fg.p_received(pe, ge, gr, lp, fle, flr)
        self.trajectoire_obj.pr_drone = pr
        self.trajectoire_pr = pr
