import unittest

from pydantic import ValidationError

from bloque01.clustering import (
    NOMBRES_CLUSTER,
    asignar_cluster,
    obtener_modelo_kmeans_final,
)
from bloque01.esquemas import DatosPaciente, DatosPacienteCluster


PACIENTE_PRUEBA = {
    "Age": 55,
    "Heart rate": 74,
    "Systolic blood pressure": 111,
    "Diastolic blood pressure": 64,
    "Blood sugar": 123,
    "CK-MB": 2.55,
    "Troponin": 0.01,
}


class ClusteringAssignmentTests(unittest.TestCase):
    def test_paciente_de_aceptacion_se_asigna_al_grupo_esperado(self):
        resultado = asignar_cluster(PACIENTE_PRUEBA)

        self.assertEqual(resultado["cluster_id"], 0)
        self.assertEqual(resultado["nombre"], NOMBRES_CLUSTER[0])
        self.assertEqual(resultado["pacientes_en_grupo"], 471)
        self.assertEqual(resultado["pct_infarto_en_grupo"], 43.52)
        self.assertEqual(
            set(resultado["perfil"]),
            {
                "edad",
                "pulso",
                "sistolica",
                "diastolica",
                "glucosa",
                "ckmb_mediana",
                "troponina_mediana",
            },
        )

    def test_modelo_final_y_perfil_se_reutilizan(self):
        modelo, escalador, perfil = obtener_modelo_kmeans_final()
        modelo_cache, escalador_cache, perfil_cache = obtener_modelo_kmeans_final()

        self.assertIs(modelo, modelo_cache)
        self.assertIs(escalador, escalador_cache)
        self.assertIs(perfil, perfil_cache)
        self.assertEqual(
            {int(cluster): int(cantidad) for cluster, cantidad in perfil["pacientes"].items()},
            {0: 471, 1: 321, 2: 240, 3: 166, 4: 116},
        )

    def test_solo_el_esquema_de_clustering_exige_marcadores_positivos(self):
        paciente_sin_marcadores = {**PACIENTE_PRUEBA, "CK-MB": 0, "Troponin": 0}
        self.assertEqual(DatosPaciente(**paciente_sin_marcadores).ck_mb, 0)

        with self.assertRaises(ValidationError) as error:
            DatosPacienteCluster(**paciente_sin_marcadores)

        mensaje = str(error.exception)
        self.assertIn("CK-MB debe ser mayor que 0", mensaje)
        self.assertIn("La troponina debe ser mayor que 0", mensaje)


if __name__ == "__main__":
    unittest.main()
