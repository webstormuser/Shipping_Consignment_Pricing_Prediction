import logging
import os
from datetime import datetime

# Log file name
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"

# Log directory
LOG_FILE_DIR = os.path.join(os.getcwd(), "logs")

# Create folder if not available
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# Add print statements for debugging
print(f"Log file path: {LOG_FILE_PATH}")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Add a print statement to confirm logging setup
print("Logging setup completed.")

# Now continue with the rest of your script...
