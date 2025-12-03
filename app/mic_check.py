import sounddevice as sd
import numpy as np

def mic_check():
    print("\nAUDIO HARDWARE DIAGNOSTIC")
    print("--------------------------------")
    
    # List Devices
    devices = sd.query_devices()
    print("Available Devices:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")
            
    # Ask for Index
    try:
        idx = int(input("\nEnter the device index to test (e.g., 0 or 1): "))
    except ValueError:
        idx = 0
        
    print(f"\nTesting Device [{idx}]... (Press Ctrl+C to stop)")
    print("Speak into your mic. You should see bars below:")
    
    try:
        def callback(indata, frames, time, status):
            vol = np.sqrt(np.mean(indata**2))
            
            bars = "#" * int(vol * 100)
            
            # If Vol is 0.0000, your mic is MUTED or BLOCKED
            print(f"\rVol: {vol:.4f} | {bars}", end="", flush=True)

        with sd.InputStream(device=idx, channels=1, callback=callback):
            while True:
                sd.sleep(500)
                
    except KeyboardInterrupt:
        print("\n\nTest Complete.")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    mic_check()