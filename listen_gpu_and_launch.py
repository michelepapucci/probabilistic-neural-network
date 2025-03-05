import time
import subprocess
import pynvml
import sys

# Initialize NVML (NVIDIA Management Library)
pynvml.nvmlInit()

# Function to check GPU memory usage
def check_gpu_memory():
    num_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total
        free_memory = memory_info.free
        free_percentage = (free_memory / total_memory) * 100
        
        # Check if free memory is less than 70%
        if free_percentage < 70:
            return False
    return True

# Function to launch another Python script with arguments
def launch_script(script_path, args):
    # Run the script with the arguments passed
    subprocess.run(['python', script_path] + args)

# Main loop to check GPU memory every 10 seconds
def monitor_and_launch(script_path, args):
    while True:
        if check_gpu_memory():
            print("All GPUs have at least 70% memory free. Launching the script...")
            launch_script(script_path, args)
            break
        else:
            print("GPU memory is too low. Checking again in 10 seconds.")
        time.sleep(10)

if __name__ == "__main__":
    # Define the argument you need to pass to the script
    script_to_launch = "generate_responses.py"  # Replace with your script path
    arguments = ["-l", "20"]  # Define the specific argument here
    
    # Start monitoring and launch the script when conditions are met
    monitor_and_launch(script_to_launch, arguments)
