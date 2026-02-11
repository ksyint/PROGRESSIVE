class DataGenerationError(Exception):
    pass

class ImageNotFoundError(DataGenerationError):
    pass

class InvalidBBoxError(DataGenerationError):
    pass

class MaskProcessingError(DataGenerationError):
    pass

class ModelInferenceError(DataGenerationError):
    pass

class ConfigurationError(DataGenerationError):
    pass

class ValidationError(DataGenerationError):
    pass

class AnatomyMaskNotFoundError(DataGenerationError):
    pass

class UnsupportedModalityError(DataGenerationError):
    pass
