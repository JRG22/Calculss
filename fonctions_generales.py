import scipy.constants as scc
from ANFRpy.geometry import Calcul_ECEF_ENU

from ANFRpy.geometry.WGS84_to_ECEF import *

import UAS as UAS
import GroundStation as Bs
import radioStation as rS

import matplotlib.pyplot as plt

# TODO Mettre en anglais les commentaires
"""
Fonctions générales géométriques et de calcul de I/N, C/I...
"""

import Affichage as Af

plt.rcParams['axes.grid'] = True

f = 5060e6
def filtred_power_vectoriel(pe_dbm, nb_canal_e, nb_canal_r, affichage=False):
    """Calcul de la puissance filtrée avec les masques d'émission et de niveaux de blocking


    Parameters
    ----------
    pe_dbm: float
        Puissance transmise en dBm
    nb_canal_e : int[]
        Canal ou canaux de ou des émetteurs
    nb_canal_r : int []
        Canal ou canaux de ou des récepteurs
    affichage : boolean
        show  graphics of masks and filtred power
    Returns
    -------
    float[]
        Tableau des Puissances filtrées en dBm
    """
    pe = pe_dbm  # dBm
    pas = 100  # pas d'échantillonage

    offset_arr = abs(nb_canal_e - nb_canal_r)  # tableau d'offset fréquentiel entre les émetteurs
    if type(offset_arr) != int:
        offset_arr = abs(nb_canal_e - nb_canal_r).astype(int)  # ecart en nombre de canaux

    # TODO mettre en ecart de fréquences

    tab_pf = np.ones(250)
    for i in range(0, 250):
        mask_emission = UAS.masques_emission(pe, pas, i)  # Récupération du masque d'émission

        mask_reception = UAS.masques_reception(pas, i)  # Recuperation du masque d'émission
        tab_pf[i] = sum_dbm(mask_emission + mask_reception)  # On aditionne les masques en dBm
    pf = tab_pf[offset_arr]  # Selection des puissances en fonction des fréquences
    if affichage:
        mask_emission = UAS.masques_emission(pe, pas, 0)
        mask_reception = UAS.masques_reception(pas, 0)
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(mask_emission)
        ax2.plot(mask_reception)
        ax3.plot(mask_emission + mask_reception)
        ax4.plot()
        ax1.title.set_text('Masque d\'emission')
        ax2.title.set_text('Masque de réception')
        ax3.title.set_text('Masque d\'emission et de reception')
        plt.tight_layout()

        plt.grid()
        plt.show()

    return pf


def calcul_dmin(px, ge, gr, c_prot, cle, clr, f, n):
    """Retourne la distance minimale de respect du critère de protection

    <Parameters>
    ----------
    px : float
        Puissance filtrée transmise en dBm
    ge : float
        Gain d'émission en dB
    gr : float
        Gain de réception en dB
    c_prot : int
        Critère de protection I/N en dB
    cle : int
        Feeder loss émetteur en dB
    clr : int
        Feeder loss récepteur en dB
    f : float
        fréquence en hz
    n : float
        noise en dBm
    Returns
    -------
    float
        Valeur de Blocking en dB
    """
    fsl_loc = px + ge + gr - c_prot - cle - clr - n
    lbda = scc.c / f
    distance = lbda * (10 ** (fsl_loc / 20)) / (4 * np.pi)
    return fsl, distance


def p_received(px, ge, gr, lp, cle, clr, affichage=False):
    """Calcule le bilan de liaison


    Parameters
    ----------
    px : float
        Puissance filtrée transmise en dBm
    ge : float
        Canal de réception décalé fréquentiellement par rapport à nbCanal
    gr : float
        Gain de réception en dB
    lp : float
        Pertes de propagation en dB
    cle : int
        Feeder loss émetteur en dB
    clr : int
        Feeder loss récepteur en dB
    Returns
    -------
    float
        Puissance reçue en dBm

    """
    if affichage:
        print("px : ", px, "ge : ", ge, "eirp = ", px + ge, " gr : ", gr, "lp : ", lp, "cle", cle, "clr : ", clr)
    pr = px + + gr + np.subtract(gr, lp) - cle - clr
    return pr


def fsl(f, d):
    """Calcule le bilan de liaison


    Parameters
    ----------

    f : float
        fréquence en hz
    d : float
        distance en mètres
    Returns
    -------
    float
        Valeur de Blocking en dB
    """
    lbda = scc.c / f

    return 20 * np.log10(4 * np.pi * d / lbda)


def sum_dbm(p):
    """Calcule une somme de puissances en dBm


        Parameters
        ----------

        p : float []
            Puissances en dBm dans un tableau

        Returns
        -------
        float
            Somme des puissances en dBm
    """
    p = 10 ** (p / 10)
    return 10 * np.log10(np.sum(p))


def calcul_distance_lla(lla1, lla2):
    """Retourne la distance séparant les deux coordonnées WGS84 lla1 et ll2


        Parameters
        ----------

        lla1 : float[]
            Latitude, longitude, altitude de l'émetteur en degrés et mètres
        lla2 : float[]
            Latitude, longitude, altitude du récepteur en degrés et mètres
        Returns
        -------
        float
            Distance en mètres
    """

    lat1 = lla1[0]
    long1 = lla1[1]
    alt1 = lla1[2]
    lat2 = lla2[0]
    long2 = lla2[1]
    alt2 = lla2[2]
    ecef_em = Calcul_WGS84_ECEF(lat1, long1, alt1)
    ecef_rec = Calcul_WGS84_ECEF(lat2, long2, alt2)
    x1 = ecef_em[0]
    y1 = ecef_em[1]
    z1 = ecef_em[2]
    x2 = ecef_rec[0]
    y2 = ecef_rec[1]
    z2 = ecef_rec[2]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return np.squeeze(distance)


def calcul_angle_emetteur_lla(lat1, long1, alt1, lat2, long2, alt2):
    # TODO : vérifier quelle station on doit mettre ne premier dans les paramètres
    """Retourne l'angle entre le vecteur de direction de l'antenne d'origine
        et le vecteur base_station - drone


        Parameters
        ----------

        lla1 : float[]
            Latitude, longitude, altitude de l'émetteur en degrés et mètres
        lla2 : float[]
            Latitude, longitude, altitude du récepteur en degrés et mètres
        Returns
        -------
        float
            Angle en degrés
    """

    """
    lat1 = lla1[0]
    long1 = lla1[1]
    alt1 = lla1[2]
    lat2 = lla2[0]
    long2 = lla2[1]
    alt2 = lla2[2]"""
    # TODO : Revoir l'angle de la station de base
    # print("--Calcul angle emetteur--", lat1, long1, alt1, lat2, long2, alt2)
    enu_em_dir = np.array([[0], [0], [10]])  # Direction par défaut de l'antenne
    # FSD_em_dir = Calcul_ENUv_FSD(int(ENU_em_dir[0]), int(ENU_em_dir[1]), int(ENU_em_dir[2]), 0, azim1, elev1)
    fsd_em_dir = enu_em_dir

    enu_em_rec = Calcul_ECEF_ENU(lat1, long1, alt1, lat2, long2, alt2)  # Vecteur Emetteur-recepteur
    fsd_em_rec = enu_em_rec

    x1 = fsd_em_dir[0]
    y1 = fsd_em_dir[1]
    z1 = fsd_em_dir[2]

    x2 = fsd_em_rec[0]
    y2 = fsd_em_rec[1]
    z2 = fsd_em_rec[2]
    angle = np.degrees(np.arccos(
        (x1 * x2 + y1 * y2 + z1 * z2) / (np.sqrt((x1 ** 2 + y1 ** 2 + z1 ** 2) * (x2 ** 2 + y2 ** 2 + z2 ** 2)))))

    return angle





def calcul_c_drone(drone, receiver):
    """Calcul de la puissance voulue émise par la station de base

    Parameters
    ----------
    drone : UAS[]
            drone recevant le signal
    receiver : radioStation[]
            Station de base émettant le signal

    Returns
    -------
    float[]
        tableau de puissance reçue voulue par le drone en dBm
    float[]
        Coordonnees du drone en réception
    """
    print("    -->Getting parameters...")
    p_uas, gr, fle, canal_c = np.vectorize(rS.radioStation.get_parameters)(drone)
    p_bs, gr, flr, canal_b = np.vectorize(rS.radioStation.get_parameters)(receiver)
    coord_reception = np.vectorize(rS.radioStation.get_coordinates)(drone)
    coord_emission = np.vectorize(rS.radioStation.get_coordinates)(receiver)
    print("    -->Coord reception", coord_reception, "coord Emission", coord_emission)
    print("    -->Calcul de la puissance voulue émise par la station de base")

    # angle_emission = calcul_angle_emetteur_lla(coord_emission, coord_reception)
    angle_emission = np.vectorize(calcul_angle_emetteur_lla)(coord_emission[0], coord_emission[1], coord_emission[2],
                                                             coord_reception[0], coord_reception[1], coord_reception[2])
    print("        -->Angle d'émission", angle_emission)
    ge = Bs.get_gain_from_angle(angle_emission)  # dB
    print("        -->Gain de l'antenne au sol", ge)
    distance = calcul_distance_lla(coord_emission, coord_reception)  # degrees

    lp = fsl(f, distance)  # dB
    print("        --> Distance drone station de base : ", distance)
    print("        --> Pertes de propagation: ", lp)
    pe = filtred_power_vectoriel(p_bs, canal_b, canal_c)  # dBm
    print("        --> Puissance filtrée ", pe)
    pr = p_received(pe, ge, gr, lp, fle, flr)
    print("        --> Puissance reçue: ", pr)

    return pr


def calcul_i_drone(drone_v, drone_i, plot_cdf=False, console=False):
    """Calcul de la puissance non voulue émise par le drone interférent

        Parameters
        ----------

        Returns
        -------
        float
            Puissance reçue non voulue par le drone
    """

    p_uas_b, gr, flr, canal_v = np.vectorize(rS.radioStation.get_parameters)(drone_v)
    p_uas_i, ge, fle, canal_i = np.vectorize(rS.radioStation.get_parameters)(drone_i)

    coord_emission = np.vectorize(rS.radioStation.get_coordinates)(drone_i)
    coord_reception = np.vectorize(rS.radioStation.get_coordinates)(drone_v)

    distance = calcul_distance_lla(coord_emission, coord_reception)

    lp = fsl(f, distance)

    if np.size(p_uas_i) > 1:
        p_uas_i = p_uas_i[0]
        print("On entre dans la boucle, p_uas_i = ", p_uas_i)
    # print("On sort dans la boucle, p_uas_i = ", p_uas_i)

    # pe = filtred_power_vectoriel(p_uas_i, canal_i, canal_v)
    pe = p_uas_i
    pr = p_received(pe, ge, gr, lp, fle, flr)

    if console:
        print("    -->Calcul de la puissance non voulue émise par un ou plusieurs drones interférents")
        print("        --> Canal utilisé par le drone : ", canal_i)
        print("        --> Distance drone station de base : ", distance)
        print("        --> Pertes de propagation: ", lp)
        print("        --> Puissance filtrée ", pe)
        print("        --> Puissance reçue: ", pr)
    if plot_cdf:
        Af.plot_cdf(distance, "ECDF Distances entre l'émission et la réception", 0, 0)
        Af.plot_cdf(pe, "ECDF Puissance filtrée", 1, 0)
        Af.plot_cdf(lp, "ECDF Propagation Loss", 0, 1)
        Af.plot_cdf(pr, "ECDF Puissance reçue", 1, 1)

    return pr


def add_i_trajectoire2(drone_v: UAS, drone_i):
    """Calcul de la puissance non voulue émise par le drone interférent reçue par le drone victime

            Parameters
            ----------

            Returns
            -------
            float
                Puissance reçue non voulue par le drone
        """

    p_uas_i, ge, fle, canal_v = np.vectorize(UAS.UAS.get_parameters)(drone_i)
    p_uas_v, gr, flr, canal_v = np.vectorize(UAS.UAS.get_parameters)(drone_v)
    pr = np.zeros(np.size(drone_v.trajectoire_obj.temps))
    tmax = min(np.size(drone_v.trajectoire_obj.temps), np.size(drone_i.trajectoire_obj.temps))
    for i in range(tmax):
        coord_reception = np.vectorize(UAS.UAS.get_coordinates)(drone_v, i)
        coord_emission = np.vectorize(UAS.UAS.get_coordinates)(drone_i, i)

        distance = calcul_distance_lla(coord_emission, coord_reception)
        drone_v.trajectoire_obj.distance_Interferer[i] = distance
        lp = fsl(f, distance)
        pe = p_uas_i
        pr[i] = p_received(pe, ge, gr, lp, fle, flr)
        drone_v.trajectoire_obj.NplusI[i] = sum_dbm(
            np.array([drone_v.trajectoire_obj.NplusI[i], pr[i]]))
    return pr


def calcul_i_trajectoire(drone_v, p_uas_i, ge, fle, canal_i, coord_emission, plot_cdf=False, console=False):
    """Calcul de la puissance non voulue émise par le drone interférent

        Parameters
        ----------

        Returns
        -------
        float
            Puissance reçue non voulue par le drone
    """

    p_uas_b, gr, flr, canal_v = np.vectorize(UAS.UAS.get_parameters)(drone_v)

    coord_reception = np.vectorize(UAS.UAS.get_coordinates)(drone_v)
    distance = calcul_distance_lla(coord_emission, coord_reception)

    lp = fsl(f, distance)
    print("        --> Calcul de la puissance filtrée")
    pe = filtred_power_vectoriel(p_uas_i, canal_i, canal_v)
    print("        --> Calcul de la puissance reçue")
    pr = p_received(pe, ge, gr, lp, fle, flr)

    if console:
        print("    -->Calcul de la puissance non voulue émise par un ou plusieurs drones interférents")
        print("        --> Canal utilisé par le drone : ", canal_i)
        print("        --> Distance drone station de base : ", distance)
        print("        --> Pertes de propagation: ", lp)
        print("        --> Puissance filtrée ", pe)
        print("        --> Puissance reçue: ", pr)
    if plot_cdf:
        Af.plot_cdf(distance, "ECDF Distances entre l'émission et la réception", 0, 0)
        Af.plot_cdf(pe, "ECDF Puissance filtrée", 1, 0)
        Af.plot_cdf(lp, "ECDF Propagation Loss", 0, 1)
        Af.plot_cdf(pr, "ECDF Puissance reçue", 1, 1)


def calcul_c_trajectoire(drone, base_station, sens):
    if sens == "up":
        p_uas, ge, fle, canal = drone.get_parameters()
        p_bs, gr, flr, canal_i = base_station.get_parameters()

        distance = calcul_distance_lla(drone.trajectoire, base_station.get_coordinates())

        # distance = calcul_distance_grand_cercle(np.radians(drone.trajectoire),np.raidans(base_station.get_coordinates))
        lp = fsl(f, distance)
        pe = p_uas
        lla_base_station = base_station.get_coordinates()
        lla_drone = drone.trajectoire
        angle_emission = drone.trajectoire_obj.angle_ground_station

        """angle_emission = np.vectorize(calcul_angle_emetteur_lla)(lla_drone[0], lla_drone[1], lla_drone[2], lla_base_station[0],
                                                                 lla_base_station[1], lla_base_station[2])"""

        ge = base_station.set_gain_from_angle(angle_emission)
        pr = p_received(pe, ge, gr, lp, fle, flr)
        drone.trajectoire_obj.pr_drone = pr
        drone.trajectoire_pr = pr

    if sens == "down":
        p_uas, ge, fle, canal = base_station.get_parameters()
        p_bs, gr, flr, canal_i = drone.get_parameters()
        distance = calcul_distance_lla(drone.trajectoire, base_station.get_coordinates())
        lp = fsl(f, distance)
        pe = p_uas
        drone.trajectoire_pr = p_received(pe, ge, gr, lp, fle, flr)


def calcul_distance_grand_cercle(uav_coord_wgs84_rad_entry, uav_coord_wgs84_rad_exit):
    a_m = 6378137 + uav_coord_wgs84_rad_entry[2]  # demi grand axe en mètre*

    lat1 = uav_coord_wgs84_rad_entry[0]
    lat2 = uav_coord_wgs84_rad_exit[0]
    long1 = uav_coord_wgs84_rad_entry[1]
    long2 = uav_coord_wgs84_rad_exit[1]
    """distance_grand_cercle_m_entry_exit = a_m * np.cos(np.radians(
                np.sin(np.radians(uav_coord_wgs84_deg_exit[0])) *
                np.sin(np.radians(uav_coord_wgs84_deg_entry[0])) + np.cos(np.radians(
                    uav_coord_wgs84_deg_exit[0])) * np.cos(np.radians(uav_coord_wgs84_deg_entry[0])) * np.cos(
                    np.radians(uav_coord_wgs84_deg_exit[1] - uav_coord_wgs84_deg_entry[1]))))"""
    d = 2 * a_m * np.arcsin(
        np.sqrt(np.sin((lat2 - lat1) / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * np.sin((long2 - long1) / 2) ** 2)
    return d


def calcul_pr_radiostations(radiostation_A, radiostation_B):
    """Calcul de la puissance reçue entre deux radiostations.fd

        Parameters
        ----------
        radiostation_b : radiostation
                drone recevant le signal
        radiostation_a : radioStation[]
                Station de base émettant le signal

        Returns
        -------
        float[]
            tableau de puissance reçue voulue par le drone en dBm
        """
    pr = np.zeros(int(np.size(radiostation_A.trajectoire_obj.temps)))
    p_1, gr, fle, canal_c = radiostation_A.get_parameters()
    p_2, gr, flr, canal_b = radiostation_B.get_parameters()
    coord_reception = radiostation_B.get_coordinates()
    for i in range(int(np.size(radiostation_A.trajectoire_obj.temps))):

        coord_emission = radiostation_A.trajectoire_obj.coordonnees[:,i]
        # angle_emission = calcul_angle_emetteur_lla(coord_emission, coord_reception)
        angle_emission = np.vectorize(calcul_angle_emetteur_lla)(coord_emission[0], coord_emission[1],
                                                                 coord_emission[2],
                                                                 coord_reception[0], coord_reception[1],
                                                                 coord_reception[2])
        ge = Bs.get_gain_from_angle(angle_emission)  # dB
        distance = calcul_distance_lla(coord_emission, coord_reception)  # degrees

        lp = fsl(f, distance)  # dB

        #pe = filtred_power_vectoriel(p_1, canal_b, canal_c)  # dBm
        pe =p_1
        pr[i] = p_received(pe, ge, gr, lp, fle, flr)






    return pr

