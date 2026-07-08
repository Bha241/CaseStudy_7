import os
import urllib.request
import zipfile
import socket
import shutil

# Set socket timeout to 30 seconds
socket.setdefaulttimeout(30)

def download_and_extract_jre():
    url = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jre_x64_windows_hotspot_17.0.10_7.zip"
    dest_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(dest_dir, "openjdk17.zip")
    extract_path = os.path.join(dest_dir, "jdk17_temp")
    final_jre_path = os.path.join(dest_dir, "jre-17")
    
    # Cleanup any previous partial downloads
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass
    if os.path.exists(extract_path):
        try:
            shutil.rmtree(extract_path)
        except Exception:
            pass
            
    if os.path.exists(final_jre_path):
        print(f"Java 17 JRE already exists at: {final_jre_path}")
        return
        
    print(f"Downloading OpenJDK JRE 17 from {url}...")
    try:
        def report_hook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = (read_so_far * 100) / total_size
                print(f"Downloaded {percent:.2f}%", end="\r")
            else:
                print(f"Downloaded {read_so_far} bytes", end="\r")
                
        urllib.request.urlretrieve(url, zip_path, report_hook)
        print("\nDownload complete.")
        
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # Find extracted folder and rename to jre-17
        extracted_folders = os.listdir(extract_path)
        if len(extracted_folders) > 0:
            src_folder = os.path.join(extract_path, extracted_folders[0])
            os.rename(src_folder, final_jre_path)
            print(f"Extracted JRE 17 to: {final_jre_path}")
            
        # Cleanup
        os.remove(zip_path)
        shutil.rmtree(extract_path)
        print("Cleanup completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass

if __name__ == "__main__":
    download_and_extract_jre()
