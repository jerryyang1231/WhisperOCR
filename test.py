import torchaudio

path = '/share/nas169/jerryyang/corpus/mandarin/clean/cv/SSB07370296.wav'

audio, _ = torchaudio.load(path)


audio = audio.transpose(0, 1)
audio = audio.squeeze(1)

print("audio :", audio)
print("audio's shape :", audio.shape)