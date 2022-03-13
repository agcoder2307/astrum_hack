from inference.mrcnn.config import Config


class PredictionConfig(Config):
    NAME = 'tumor_cfg'
    NUM_CLASSES = 1 + 6
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
