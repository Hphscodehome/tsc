import os
from datetime import datetime
def get_unique_log_dir(base_dir="./logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # 例如：20250222_143022
    log_dir = os.path.join(base_dir, f"run_{timestamp}")
    return log_dir
