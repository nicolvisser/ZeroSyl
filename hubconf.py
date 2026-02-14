dependencies = ["torch", "faiss", "numpy", "scipy"]


from zerosyl import LanguageModel, ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete


def zerosyl_collapsed() -> ZeroSylCollapsed:
    return ZeroSylCollapsed.from_remote()


def zerosyl_discrete() -> ZeroSylDiscrete:
    return ZeroSylDiscrete.from_remote()


def zerosyl_continuous() -> ZeroSylContinuous:
    return ZeroSylContinuous.from_remote()


def language_model() -> LanguageModel:
    return LanguageModel.from_remote()
