import enum

import gin


@gin.constants_from_enum
@enum.unique
class TevaTasks(enum.Enum):
    WURA = "wura"
    IFT = "instruction_finetuning"
    SFT = "supervised_finetuning"
    EVAL = "eval"
    # ----
    # EVAL
    # ----
    MASAKHANEWS = "masakhanews"
    LAFAND = "lafand"
    XLSUM = "xlsum"
    SQUAD = "squad"
    AFRIQA = "afriqa"
    # NAIJARC = "naijarc"
    # BELEBELE = "belebele"
    # SIB = "sib"
    # ---
    # IFT
    # ---
    HUMAN_AYA = "aya-dataset"
    TRANSLATED_AYA = "translated_aya"
    TEMPLATED_AYA = "templated_aya"
    AYA_COLLECTION = "aya_collection"
    XP3X = "xp3x"
    OCTOPACK_OSST = "octopack_osst"
    OIG_SMALL_CHIP2 = "oig_small_chip2"
    TASKSOURCE_INSTRUCT = "tasksource_instruct"
    FLAN_NIV2_SUBMIX = "flan_niv2_submix"
    FLAN2021_SUBMIX = "flan2021_submix"
    FLAN_COT_SUBMIX = "flan_cot_submix"
    FLAN_DIALOG_SUBMIX = "flan_dialog_submix"
    FLAN_T0_SUBMIX = "flan_t0_submix"
    FLAN_COLLECTION = "flan_collection"
    DPI_TEMPLATED = "dpi_templated"
    TEMPLATED_IFT = "templated_ift"

    @classmethod
    def get_sub_tasks(cls, mixture: "TevaTasks", **kwargs) -> frozenset["TevaTasks"]:
        SUBTASKS_MAP: dict["TevaTasks", callable] = {
            cls.AYA_COLLECTION: cls.get_aya_collection_tasks,
            cls.DPI_TEMPLATED: cls.get_dpi_templated_tasks,
            cls.FLAN_COLLECTION: cls.get_flan_collection_tasks,
            cls.IFT: cls.get_instruction_tasks,
            cls.SFT: cls.get_supervised_ft_tasks,
            cls.TEMPLATED_IFT: cls.get_templated_instruction_tasks,
        }

        return SUBTASKS_MAP[mixture](**kwargs)
    
    @classmethod
    def get_supervised_ft_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.LAFAND, cls.MASAKHANEWS, cls.XLSUM, cls.SQUAD
        ])
    
    @classmethod
    def get_evaluation_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.LAFAND, cls.MASAKHANEWS, cls.XLSUM, cls.AFRIQA
        ])
    
    @classmethod
    def get_flan_collection_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([
            cls.FLAN2021_SUBMIX,
            cls.FLAN_COT_SUBMIX,
            cls.FLAN_DIALOG_SUBMIX,
            cls.FLAN_NIV2_SUBMIX,
            cls.FLAN_T0_SUBMIX
        ])
    
    @classmethod
    def get_dpi_templated_tasks(cls, flatten_flan_collection: bool = False) -> frozenset["TevaTasks"]:
        tasks = [cls.TASKSOURCE_INSTRUCT, cls.OIG_SMALL_CHIP2, cls.OCTOPACK_OSST]

        if flatten_flan_collection:
            tasks += cls.get_flan_collection_tasks()
        else:
            tasks.append(cls.FLAN_COLLECTION)
        
        return frozenset(tasks)
    
    @classmethod
    def get_aya_collection_tasks(cls) -> frozenset["TevaTasks"]:
        return frozenset([cls.HUMAN_AYA, cls.TEMPLATED_AYA, cls.TRANSLATED_AYA])
    
    @classmethod
    def get_templated_instruction_tasks(cls, flatten_mixtures: bool = False) -> frozenset["TevaTasks"]:
        tasks = [cls.XP3X, cls.TEMPLATED_AYA]
        
        if flatten_mixtures:
            tasks += cls.get_dpi_templated_tasks(flatten_flan_collection=True)
        else:
            tasks.append(cls.DPI_TEMPLATED)

        return frozenset(tasks)
    
    @classmethod
    def get_instruction_tasks(cls) -> frozenset["TevaTasks"]:
        return cls.get_templated_instruction_tasks(flatten_mixtures=True) | cls.get_aya_collection_tasks()
