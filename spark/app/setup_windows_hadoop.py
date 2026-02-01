"""
Helper script to download winutils.exe for Windows Spark development
"""
import os
import urllib.request
import zipfile
import tempfile
import shutil

def setup_winutils():
    """Download and setup winutils.exe for Windows"""
    if os.name != 'nt':
        print("This script is only for Windows")
        return False
    
    # Create hadoop directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hadoop_home = os.path.join(script_dir, 'hadoop')
    bin_dir = os.path.join(hadoop_home, 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    
    winutils_path = os.path.join(bin_dir, 'winutils.exe')
    
    if os.path.exists(winutils_path):
        print(f"[OK] winutils.exe already exists: {winutils_path}")
        return hadoop_home
    
    print("Downloading winutils.exe...")
    print("Note: This may take a few minutes...")
    
    # Download winutils from a reliable source
    # Using a direct link to winutils.exe (Hadoop 3.2.0)
    winutils_url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.2.0/bin/winutils.exe"
    
    try:
        urllib.request.urlretrieve(winutils_url, winutils_path)
        print(f"[OK] Successfully downloaded winutils.exe to: {winutils_path}")
        print(f"\nSet these environment variables:")
        print(f"  HADOOP_HOME={hadoop_home}")
        print(f"  hadoop.home.dir={hadoop_home}")
        return hadoop_home
    except Exception as e:
        print(f"[ERROR] Error downloading winutils.exe: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://github.com/cdarlint/winutils")
        print(f"  Place winutils.exe in: {bin_dir}")
        return None

if __name__ == "__main__":
    hadoop_home = setup_winutils()
    if hadoop_home:
        print(f"\n[OK] Setup complete! HADOOP_HOME: {hadoop_home}")
        print("You may need to restart your Python session for changes to take effect.")
