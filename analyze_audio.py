import soundfile as sf
import numpy as np

audio, sr = sf.read('my_generated_song.wav')

chunk_size = sr * 10
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

print("Analyzing different time segments:")
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i} ({i*10}-{(i+1)*10}s):")
    print(f"  mean: {chunk.mean():.6f}")
    print(f"  std: {chunk.std():.6f}")
    print(f"  unique values: {len(np.unique(chunk))}")
    print(f"  energy: {np.mean(chunk**2):.8f}")

print(f"\nFull audio:")
print(f"  mean: {audio.mean():.6f}")
print(f"  std: {audio.std():.6f}")
print(f"  unique values: {len(np.unique(audio))}")
print(f"  energy: {np.mean(audio**2):.8f}")

# Check if audio is repeating
print("\nChecking for repetition patterns:")
for i in range(0, min(5, len(chunks)-1)):
    similarity = np.corrcoef(chunks[i].flatten(), chunks[i+1].flatten())[0,1]
    print(f"  Chunk {i} vs Chunk {i+1}: correlation = {similarity:.6f}")
