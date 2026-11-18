from pydantic import BaseModel, Field, field_validator


class DatosPaciente(BaseModel):
    edad: float = Field(alias="Age", gt=0, le=120)
    frecuencia_cardiaca: float = Field(alias="Heart rate", gt=0, le=300)
    presion_sistolica: float = Field(alias="Systolic blood pressure", gt=0, le=350)
    presion_diastolica: float = Field(alias="Diastolic blood pressure", gt=0, le=250)
    glucosa: float = Field(alias="Blood sugar", gt=0, le=2000)
    ck_mb: float = Field(alias="CK-MB", ge=0)
    troponina: float = Field(alias="Troponin", ge=0)


class DatosPacienteCluster(DatosPaciente):
    ck_mb: float = Field(alias="CK-MB")
    troponina: float = Field(alias="Troponin")

    @field_validator("ck_mb")
    @classmethod
    def validar_ck_mb_positivo(cls, valor: float) -> float:
        if not valor > 0:
            raise ValueError("CK-MB debe ser mayor que 0 para calcular el logaritmo.")
        return valor

    @field_validator("troponina")
    @classmethod
    def validar_troponina_positiva(cls, valor: float) -> float:
        if not valor > 0:
            raise ValueError("La troponina debe ser mayor que 0 para calcular el logaritmo.")
        return valor


class PerfilCluster(BaseModel):
    edad: float
    pulso: float
    sistolica: float
    diastolica: float
    glucosa: float
    ckmb_mediana: float
    troponina_mediana: float


class AsignacionClusterResponse(BaseModel):
    cluster_id: int
    nombre: str
    pacientes_en_grupo: int
    pct_infarto_en_grupo: float
    perfil: PerfilCluster
