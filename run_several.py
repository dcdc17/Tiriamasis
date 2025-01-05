import subprocess

def execute_python_files(files):
    """
    Execute a list of Python files.
    Args:
        files (list): List of file paths to execute.
    """

    for file in files:
        # Execute the Python file
        subprocess.run(['python3', file])


# List of .py files to execute
analysis_python_files = ['AB.py', 'Clustering.py', 'Corr.py', 'Granger.py', 'Regime.py']
forecast_python_files = ['VARMAX.py', 'GARCH.py', 'LSTM_GRU.py']

# Run the scripts
execute_python_files(analysis_python_files)
execute_python_files(forecast_python_files)