# ==================================================
# The following run script is used when your system based on Unix (Linux, MacOS, ...)
# If training and testing perform on 2 different system, make sure params.ENV is set to training environment
# and perform just one process at single time
#
# ==================================================

# Training and Validation
python3 main.py

# testing
python3 test.py