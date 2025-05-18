import subprocess
import sys

def install_ffmpeg():
    print("Starting FFmpeg installation...")

    # pip installation + upgrade
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "--upgrade", "pip"])
    
    # setuptools installation + upgrade
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "setuptools"])

    try:
        # FFmpeg installation
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "ffmpeg-python"])
        print("FFmpeg installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing ffmpeg-python via pip: {e}")


    try:
        # FFmpeg static binary installation
        print("Installing FFmpeg (Static Binary)...")
        subprocess.check_call(["wget", "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz", 
                               "-O", "/tmp/ffmpeg.tar.xz"])
        
        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", 
                               "-C", "/tmp/"])
        
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"], 
            capture_output = True, 
            text = True
        )

        ffmpeg_path = result.stdout.strip()

        # -- Binary -->
        # Copy the ffmpeg binary to /usr/local/bin
        # Make it executable
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
        
        print("FFmpeg (Static Binary) installed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error installing FFmpeg (Static Binary): {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please install wget and tar.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("FFmpeg installation process completed.")

    try:
        # Check FFmpeg version
        print("Checking FFmpeg version...")
        # Check if ffmpeg is installed
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output = True, 
            text = True,
            check = True
        )

        print("==========================")
        print("FFmpeg version:")
        print("==========================")
        print(result.stdout)
        print("==========================")
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:        
        print("FFmpeg Verification Failed...!")
        print(f"Error checking FFmpeg version: {e}")
        return False
