import pygetwindow as gw
import time

def arrange():
    print("Finding windows...")
    lookcam = None
    chrome = None
    
    # Try to find the windows
    for w in gw.getAllWindows():
        if not w.title: continue
        title = w.title.lower()
        
        # Find LookCam (excluding Chrome if it has LookCam in the title)
        if 'lookcam' in title and 'chrome' not in title:
            lookcam = w
            
        # Find Chrome
        if 'chrome' in title:
            chrome = w

    if not lookcam:
        print("Could not find the LookCam window. Is it open?")
    else:
        print(f"Found LookCam window: {lookcam.title}")
        try:
            lookcam.restore()  # Un-maximize if it is maximized
            lookcam.moveTo(0, 0)
            lookcam.resizeTo(900, 800)
            print("LookCam window moved to the left.")
        except Exception as e:
            print(f"Failed to move LookCam: {e}")

    if not chrome:
        print("Could not find the Google Chrome window.")
    else:
        print(f"Found Chrome window: {chrome.title}")
        try:
            chrome.restore()  # Un-maximize if it is maximized
            chrome.moveTo(900, 0)
            chrome.resizeTo(900, 800)
            print("Chrome window moved to the right.")
        except Exception as e:
            print(f"Failed to move Chrome: {e}")

if __name__ == '__main__':
    arrange()
