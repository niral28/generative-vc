# Exploring Voice Conversion Using Generative Approaches

Niral Shah

Stanford CS230 Deep Learning 

Professor Andrew Ng

## Content:
This repo contains several Voice Conversion Techniques tried using generative deep learning methods: 

- `/autovc` 
    - An autoencoder-decoder Voice Conversion Technique using [AutoVC](https://arxiv.org/abs/1905.05879)
    
    - Pytorch Implementation forked off https://github.com/auspicious3000/autovc

    - Training on VoxCeleb1 dataset for Zero Shot Voice Conversion

    - Download [model checkpoints](https://drive.google.com/drive/folders/1lumwj3ijr0SMvWGWo-HM_RjgNUHiqGnQ?usp=sharing)

-   `audio_neural_style_transfer_vgg19.py`
    - Re-implemented [Neural Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) using pre-trained VGG19 but using Mel-Spectrogram for Audio Style Transfer
    - See `final_ast.wav ` for sample after 1000 iterations (very noisy). 

    - Results after 1000 iterations:
        - Input:
            - ![Final ](mel_spectrogram_final_ast_vgg19.png?raw=true "Input")
        - Generated:
            - ![Final ](final_generated_vgg19.png?raw=true "Title")


- `audio_neural_style_transfer_vggish.py` 
    - Similarly applied the Neural Style Transfer algorithm but used a pretrained VGGish (audioset weights)

    - See `vggish_sample.wav` for sample after 1000 iterations (also very noisy). 

    - Results after 1000 iterations:
        - Input:
            - ![Input Spectrogram](vggish_features.png?raw=true "Input")
        - Generated:
            - ![Final ](final_vggish_melspectrogram.png?raw=true "Title")

## Next Steps:

- As feature extractors VGG19 and VGGish both fall short for the Neural Style Transfer algorithm producing very noisy results. So next possible steps include training my own shallow conv_net as an audio classifier on voice and applying a different method in calculating style. 

- The current autovc implementation provided is in PyTorch, to create consistency I plan on re-implementing it in Tensorflow. 


