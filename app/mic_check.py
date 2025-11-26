import sounddevice as sd
import numpy as np

def mic_check():
    print("\n🎤 AUDIO HARDWARE DIAGNOSTIC")
    print("--------------------------------")
    
    # 1. List Devices
    devices = sd.query_devices()
    print("Available Devices:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")
            
    # 2. Ask for Index
    try:
        idx = int(input("\nEnter the device index to test (e.g., 0 or 1): "))
    except ValueError:
        idx = 0
        
    print(f"\n🎧 Testing Device [{idx}]... (Press Ctrl+C to stop)")
    print("Speak into your mic. You should see bars below:")
    
    try:
        def callback(indata, frames, time, status):
            # Calculate volume (Root Mean Square)
            vol = np.sqrt(np.mean(indata**2))
            
            # Create a visual bar
            bars = "#" * int(vol * 100)
            
            # Print status
            # If Vol is 0.0000, your mic is MUTED or BLOCKED
            print(f"\rVol: {vol:.4f} | {bars}", end="", flush=True)

        with sd.InputStream(device=idx, channels=1, callback=callback):
            while True:
                sd.sleep(500)
                
    except KeyboardInterrupt:
        print("\n\n✅ Test Complete.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    mic_check()