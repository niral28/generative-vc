
# coding: utf-8

# In[9]:


import soundfile as sf
import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen
    
spect_vc = pickle.load(open('results_test_final.pkl', 'rb'))
use_cuda = torch.cuda.is_available()
print('Is cuda available?', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth",map_location=device)
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    waveform = wavegen(model, c=c)
    sf.write('/data/results/'+name+'.wav', waveform, 16000)

