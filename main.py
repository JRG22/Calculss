import sys
import fonctions_generales as fg
import Affichage as Af
import UAS as UAS
import GroundStation as Bs
import Satellite as St
import radioStation as rS
import numpy as np
import time
import matplotlib.pyplot as plt
import export_management as em

import cartographie as cg

sys.path.append('../')

plt.rcParams['axes.grid'] = True

'''=============================================================================
 |
 |       Author:  Julien Roussel-Galle
 |       Date: 21/06/2022           
 +-----------------------------------------------------------------------------
 |
 |  Description:  Main
 |
 |       
 |
 *===========================================================================*/
'''


# TODO Rajouter commentaires + mettre en anglais

def setup(drone, satellite, ground_station, tab_drone_i):
    em.clear_file()
    angle_arrivee_drone = np.random.randint(0, 365)
    drone.set_trajectoire_bs(ground_station.get_coordinates(), angle_arrivee_drone)
    drone.limite_c_i = 10
    drone.limite_c_i = 10
    drone.limite_c_i = 10
    drone.limite_i_n = -6
    drone.limite_i_n = -6
    drone.limite_i_n = -6
    for i in range(np.size(tab_drone_i)):
        angle_arrivee_drone = np.random.randint(0, 365)
        drone.set_trajectoire_bs(ground_station.get_coordinates(), angle_arrivee_drone)
        drone_i = UAS.UAS(2, 2, 500)
        tab_drone_i[i] = drone_i
        tab_drone_i[i].set_trajectoire_bs(ground_station.get_coordinates(), angle_arrivee_drone)
    print("main : ", tab_drone_i[0])
    return drone, satellite, ground_station, tab_drone_i
def calcul_criterias(drone, satellite, ground_station):
    drone.calcul_criteria_c_i()
    drone.calcul_criteria_i_n()
    ground_station.calcul_criteria_c_i()
    ground_station.calcul_criteria_i_n()
    satellite.calcul_criteria_c_i()
    satellite.calcul_criteria_i_n()


def traj_rectiligne_altitude_constante():
    """ Crée un scénario avec des drones à trajectoire rectiligne, altitude constante. Resultats du scénario exportés en csv.
        Affichage du scénario en 2D via matplotlib


        Parameters
        ----------
        Returns
        -------

        """
    print(" _______________Scénario Trajectoire rectiligne Altitude constante drone seul _______________")
    # Creating objects ...
    satellite = St.Satellite(48, 2, 800)
    ground_station = Bs.BS(48, 2, 30)
    drone = UAS.UAS(2, 2, 500)
    nb_drones_interf = 2
    tab_drone_i = np.empty(nb_drones_interf, dtype=UAS.UAS)
    # Setup/configuration des objets ...
    drone, satellite, ground_station, tab_drone_i = setup(drone, satellite, ground_station, tab_drone_i)
    calcul_interf_recue = True #Veut on calculer l'interférence

    # puissances_recues = fg.calcul_c_trajectoire(drone, ground_station, "up")

    if calcul_interf_recue:
        print("main: ", tab_drone_i[0])
        drone.calcul_interf(tab_drone_i)
        # TODO : distinguer les deux puissances terrestrial et satellitaires
        drone.calcul_C_terrestrial(ground_station)
        drone.calcul_C_satellite(satellite)
        satellite.calcul_C(drone)
        satellite.calcul_interf(tab_drone_i)
        ground_station.calcul_interf(tab_drone_i)
        # drone.add_i_trajectoire2(drone_i)
        drone.calcul_interf(tab_drone_i)
        satellite.calcul_c(drone)
        satellite.calcul_i(drone, tab_drone_i)

        ground_station.calcul_c(drone)
        ground_station.calcul_i(drone, tab_drone_i)

    drone.calcul_criterias()
    ground_station.calcul_criterias()
    satellite.calcul_criterias()

    print("main : Probab de messages perdus en uplink reçus par le drone  ", drone.get_probability_message_lost(10, 2))
    em.export_char_trajectoire(drone)
    satellite.export_char_satellite()
    ground_station.export_char_ground_station()

    em.export_summary_(drone, satellite, ground_station)

    # drone.trajectoire_obj.plot_c_surn_fct_temps()
    # cg.carte(drone, base_station, tab_drone_i)
    if False:
        cg.carte_test(drone, base_station, tab_drone_i)
    # Af.plot_cdf(puissances_recues)


def sc1():
    nb_canal_e = 0
    B = 250e3
    f = 5030e6 + nb_canal_e * B  # fréquence d'émission en hz
    nb_canal_r = np.arange(0, 10, 1)

    px = fg.filtred_power_vectoriel(40, nb_canal_e, nb_canal_r)  # puissance filtrée en dBm
    print("noise = ", rS.calcul_N_manual(7, B))
    fsl, dmin = fg.calcul_dmin(px, 3, 3, -6, 2, 2, f, rS.calcul_N_manual(7, B))
    Af.affichage_scenario1(nb_canal_r, dmin)


def sc2():
    top = time.time()
    nb_iterations = 1000000
    print(" _______________Scénario 2 : Positions des drones victimes aléatoires -> ECDF_______________")
    affichage = Af.Affichage()  # Création de l'objet affichage récupérant les méthodes d'affichage
    print("-> Nombre d'itérations : ", nb_iterations)

    # Caractéristiques de la station de base
    base_station = Bs.BS(122, 1, 1, 10)
    # Caractéristiques du drone interférent
    drone_i = UAS.UAS(6, 2, 2, 4000)
    print("--> Temps : ", time.time() - top, " s  --> Création tableau de drones ")
    tab_drone_v = np.vectorize(UAS.UAS)(np.ones(nb_iterations) * 5, 1, 1, 4000)
    print("--> Temps : ", time.time() - top, " s --> On randomize les coordonnées du drone")
    np.vectorize(rS.radioStation.set_random_coordinates)(tab_drone_v, 1, 3)
    np.vectorize(rS.radioStation.set_random_canal)(tab_drone_v)
    print("--> Temps : ", time.time() - top, " s  --> On calcule la puissance reçue voulue")
    c = fg.calcul_c_drone(tab_drone_v, base_station)
    print("--> Temps : ", time.time() - top, " s --> Calcul des interférences ")
    inter = fg.calcul_i_drone(tab_drone_v, drone_i)
    print("--> Temps : ", time.time() - top, " s --> Récupération des coordonnées des drones victimes")
    print("--> Temps : ", time.time() - top, " s  --> Calcul du critère C/I ou I/N")
    affichage.set_criteria_csuri(c, inter)
    print("affichage criteria", affichage.criteria)
    print("--> Temps : ", time.time() - top, " s  --> Affichage")
    # affichage.plot_all(True)
    affichage.plot_cdf_csuri()
    affichage.set_criteria_isurn(inter)

    affichage.plot_cdf_isurn()
    plt.tight_layout()


def sc3(pe=40):
    top = time.time()
    nb_iterations = 40000

    print(" _______________Scénario 3 : Positions des drones interférents aléatoires -> ECDF_______________")
    affichage = Af.Affichage()  # Création de l'objet affichage récupérant les méthodes d'affichage
    print("--->Puissance émise : ", pe)
    print("-> Nombre d'itérations : ", nb_iterations)

    # Caractéristiques de la station de base
    base_station = Bs.BS(122, 1, 1, 10)
    # Caractéristiques du drone interférent
    drone_v = UAS.UAS(122, 2, 2, 4000, pe)
    # Calculs
    print("--> Temps : ", round(time.time() - top, 2), " s  --> Création tableau de drones ")
    tab_drone_i = np.vectorize(UAS.UAS)(np.ones(nb_iterations) * 123, 1, 1, 4000, pe)
    print("--> Temps : ", round(time.time() - top, 2), " s --> On randomize les coordonnées du drone")
    np.vectorize(rS.radioStation.set_random_coordinates)(tab_drone_i, 1, 3)
    print("--> Temps : ", round(time.time() - top, 2), " s --> On randomize le numero de canal du drone I")
    np.vectorize(UAS.UAS.set_random_canal)(tab_drone_i)
    print("--> Temps : ", round(time.time() - top, 2), " s  --> On calcule la puissance reçue voulue")
    c = fg.calcul_c_drone(drone_v, base_station)
    print("--> Temps : ", round(time.time() - top, 2), " s --> Calcul des interférences ")
    inter = fg.calcul_i_drone(drone_v, tab_drone_i, False, False)
    print("--> Temps : ", round(time.time() - top, 2), " s --> Récupération des coordonnées des drones victimes")
    affichage.lat, affichage.long, affichage.alt = np.vectorize(UAS.UAS.get_coordinates)(tab_drone_i)
    print("--> Temps : ", round(time.time() - top, 2), " s  --> Calcul du critère C/I ou I/N")
    affichage.set_criteria_csuri(c, inter)
    print("--> Temps : ", round(time.time() - top, 2), " s  --> Affichage")
    # affichage.plot_3d(drone_v.get_coordinates(), base_station.get_coordinates())
    # affichage.plot_criteria_3d("Critère I/N")
    affichage.plot_cdf_csuri()
    affichage.set_criteria_isurn(inter)
    affichage.plot_cdf_isurn()
    plt.tight_layout()


def sc4(pe=40):
    top = time.time()
    print(" _______________Scénario 4 : Scénario trajectoire I/N pour la station de base_______________")
    affichage = Af.Affichage()  # Création de l'objet affichage récupérant les méthodes d'affichage
    nb_iterations = 50
    print("-> Nombre de trajectoires de drones : ", nb_iterations)

    base_station = Bs.BS(122, 2, 2, 50)  # Caractéristiques de la station de base
    drone_v = UAS.UAS(122, 3, 3, 4000)
    print("--> Temps : ", time.time() - top, " s  --> Création tableau de drones ")
    tab_drone_i = np.vectorize(UAS.UAS)((np.ones(nb_iterations) * round(np.random.uniform(1, 244))).astype(int), 2, 2,
                                        np.ones(nb_iterations) * round(np.random.uniform(0, 8000)))
    angle_entry_deg = np.ones(nb_iterations)
    print("--> Temps : ", time.time() - top, " s  --> Randomize altitude and channel use ")
    for i in range(np.size(angle_entry_deg)):
        tab_drone_i[i].altitude = round(np.random.uniform(10, 8000))
        angle_entry_deg[i] = np.random.uniform(0, 360)
    print('angle entry deg', angle_entry_deg)
    print("--> Temps : ", time.time() - top, " s  --> Création des trajectoires pour tout les drones  ")
    np.vectorize(UAS.UAS.set_trajectoire_bs, excluded=['coord_bs'])(tab_drone_i, coord_bs=drone_v.get_coordinates(),
                                                                    angle_entry=angle_entry_deg)

    tab_coord_i = np.array([])
    tab_canal_i = np.array([])
    print("--> Temps : ", time.time() - top, " s  --> Récupération des coordonnées, des canauax")

    for i in range(np.size(angle_entry_deg)):
        tab_coord_i = np.append(tab_coord_i, tab_drone_i[i].trajectoire)
        tab_canal_i = np.append(tab_canal_i, tab_drone_i[i].canal)
    print("--> Temps : ", time.time() - top, " s  --> Calcul de la puissance interférence reçue")
    tab_i = fg.calcul_i_trajectoire(drone_v, pe, 3, 2, tab_canal_i, tab_coord_i, False, False)
    print("--> Temps : ", time.time() - top, " s  --> Calcul de la puissance reçue voulue")
    c = fg.calcul_c_drone(drone_v, base_station)
    affichage.add_i(tab_i)
    affichage.set_criteria_csuri(c, tab_i)
    affichage.plot_cdf_csuri()
    affichage.set_criteria_isurn(affichage.i)
    affichage.plot_cdf_isurn()
    affichage.plot_3d(drone_v.get_coordinates(), base_station.get_coordinates())
    plt.show()


def sc5bestcase():
    # TODO revoir le but de ce scenario
    Te = 290 * 6
    B = 1
    N = 10 * np.log10(1.38064852E-23 * Te * B) + 30
    # Scenario avec satellite
    satellite_gso = St.Satellite(122, 1, 1)
    drone = UAS.UAS(122, 1, 1, 50, 10 * np.log10(25) + 30)
    drone.fl = 0
    satellite_gso.FL = 0.5
    C = fg.p_received(drone.pe, drone.ge, satellite_gso.GE, fg.fsl(5030e6, fg.calcul_distance_lla(
        drone.get_coordinates(), satellite_gso.get_coordinates())), drone.fl, satellite_gso.FL, True)

    print("C = ", C)
    print("N= ", N)
    print("C/N", C - N, "diff = 0", np.abs(C - N - 63.3))


def sc6():
    """
        Scénario 6 : (Voir document 2.1 CNPC Char 5G Satellite)
    """
    # TODO remodifier cette fonction avec les actuels changements
    top = time.time()
    # List of parameters
    # Range de 55 degrés de longitude
    # 0 -> 55 degrés
    canaux = np.arange(1, 8, 1)
    satellite_gso = St.Satellite(1, 1, 1, 36000000)  # Satellite
    # Characteristics of UAS
    tab_uas1 = np.vectorize(UAS.UAS)(canaux, 1, 1, 4000)  # UAS with variables channels

    # Results for each UAS
    results1 = fg.calcul_c_drone(tab_uas1, satellite_gso) - fg.calcul_N(250000, 7)  # Calcul de C/N
    print("size", np.size(results1))
    print("Noise = ", fg.calcul_N(250000, 7))
    print(results1)
    Af.affichage_scenario6(results1, canaux)


"""def graph():
    graphviz = GraphvizOutput()
    graphviz.output_file = 'basic.png'

    with PyCallGraph(output=graphviz):
        sc3()
"""

if __name__ == '__main__':
    start = time.time()

    traj_rectiligne_altitude_constante()

    end = time.time()
    elapsed = end - start
    plt.show()
    print(f'Temps d\'exécution : {elapsed} s')
