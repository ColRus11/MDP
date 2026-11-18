import unittest

from bloque01.clasificacion import RUTA_MODELO, entrenar_random_forest
from bloque01.dbscan import analizar_pacientes_atipicos
from bloque01.esquemas import DatosPaciente
from bloque01.rutas import predecir_paciente, resumen_clasificacion


PACIENTE_PRUEBA = {
    "Age": 55,
    "Heart rate": 74,
    "Systolic blood pressure": 111,
    "Diastolic blood pressure": 64,
    "Blood sugar": 123,
    "CK-MB": 2.55,
    "Troponin": 0.01,
}


class ClassificationAnalysisTests(unittest.TestCase):
    def test_entrenamiento_se_reutiliza_en_prediccion_y_resumen(self):
        primero = entrenar_random_forest()
        segundo = entrenar_random_forest()

        self.assertIs(primero, segundo)
        self.assertEqual(len(primero["modelo"].estimators_), 100)

        resumen = resumen_clasificacion()
        prediccion = predecir_paciente(
            DatosPaciente(**PACIENTE_PRUEBA),
            indice_arbol=0,
        )

        self.assertEqual(resumen["n_estimators"], 100)
        self.assertTrue(RUTA_MODELO.is_file())
        self.assertEqual(resumen["test_samples"], 263)
        self.assertEqual(resumen["configuracion"]["random_state"], 42)
        self.assertEqual(len(resumen["configuracion"]["dataset_sha256"]), 64)
        self.assertEqual(len(resumen["confusion_matrix"]), 2)
        self.assertEqual(len(resumen["feature_importances"]), 7)
        self.assertIn("positive", resumen["classification_report"])
        self.assertEqual(
            set(prediccion["probabilities"]),
            {"negative", "positive"},
        )
        self.assertAlmostEqual(sum(prediccion["probabilities"].values()), 1.0)
        explicacion = prediccion["explicacion_arbol"]
        self.assertEqual(explicacion["indice_arbol"], 0)
        self.assertEqual(explicacion["total_arboles"], 100)
        self.assertIn(explicacion["nodo_hoja"], explicacion["nodos_visitados"])
        self.assertTrue(
            next(
                nodo
                for nodo in explicacion["nodos"]
                if nodo["id"] == explicacion["nodo_hoja"]
            )["es_hoja"]
        )


class DbscanAnalysisTests(unittest.TestCase):
    def test_analisis_dbscan_refleja_perfil_real_y_parametros(self):
        resultado = analizar_pacientes_atipicos(eps=2.0, min_samples=14)

        self.assertEqual(resultado["parametros"], {"eps": 2.0, "min_samples": 14})
        self.assertEqual(resultado["total_pacientes"], 1314)
        self.assertEqual(
            resultado["pacientes_ruido"] + resultado["pacientes_densos"],
            resultado["total_pacientes"],
        )
        self.assertEqual(resultado["pacientes_ruido"], 49)
        self.assertEqual(resultado["pacientes_densos"], 1265)
        self.assertEqual(
            [grupo["grupo"] for grupo in resultado["comparacion"]],
            ["ruido", "resto"],
        )
        self.assertEqual(len(resultado["graficos"]), 4)
        self.assertIn("no necesariamente errores de medición", resultado["explicacion"])


if __name__ == "__main__":
    unittest.main()
