DATASET_NAME = "NiGuLa/Russian_Sensitive_Topics"
LABEL_COLS = [
    "offline_crime",
    "online_crime",
    "drugs",
    "gambling",
    "pornography",
    "prostitution",
    "slavery",
    "suicide",
    "terrorism",
    "weapons",
    "body_shaming",
    "health_shaming",
    "politics",
    "racism",
    "religion",
    "sexual_minorities",
    "sexism",
    "social_injustice",
]
TEXT_COL = "text"
LABEL_THRESHOLD = 0.5
RUBERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15