# HiFi-GAN LaughNet: an implementation of Luong and Yamagishi's LaughNet (2021) for a masters thesis.

Luong and Yamagishi (2021) based [LaughNet](https://arxiv.org/abs/2110.04946) on the [HiFi-GAN](https://arxiv.org/abs/2010.05646) speech synthesis model from Kong, Kim, and Bae (2020).<br/>
The major change they applied was the substitution of spectrograms for [waveform-silhouettes](https://github.com/nii-yamagishilab/waveform-silhouette-module).<br/>
For my masters thesis I have implemented this change so I could use LaughNet to synthesise my own laughter samples to perform research with.

**Abstract :**



## Pre-requisites
1. Python >= 3.9.5
2. Clone this repository.
3. Clone the [vctk-silence-labels](https://github.com/nii-yamagishilab/vctk-silence-labels) repository inside this repository.
4. Install the Python requirements. Please refer to [requirements.txt](requirements.txt)
5. Download the [VCTK dataset](https://datashare.ed.ac.uk/handle/10283/3443) and extract the VCTK-Corpus-0.92.zip folder in this repository. 

## Preprocessing
1. Preprocess VCTK for training
	```
	python preprocessing.py --data VCTK
	```
2. Preprocess the source laughter for finetuning
	```
	python preprocessing.py --data laughter
	```

## Training
```
python train.py --config config_v1.json --input_wavs_dir VCTK-0.92/wavs --input_training_file VCTK-0.92/training.txt --input_validation_file VCTK-0.92/validation.txt
```

Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

General loss total during training with V1 generator.<br>
![General loss total](./GLT.png)
The rising trend in the General loss total is not surprising given the lossy format of the min-max nature of the waveform-silhouette compared to the original mel-spectrogram.

Mel-spectrogram error during training with V1 generator.<br>
![Mel-spectrogram error](./MSE.png)

Validation mel-spectrogram error during training with V1 generator.<br>
![validation Mel-spectrogram error](./VMSE.png)

## Fine-Tuning
1. Extract waveform silhouettes from the source laughter in numpy format using the following command:
    ```
    python extract_ws_tensors.py
    ```
2. Copy the filename of the source laughter file you want to finetune on from the training-ft.txt file in the laughter/output directory (including the | token!) to the validation-ft.txt file.
3. Fine-tune on the source laughter using the following command: 
    ```
    python train.py --fine_tuning True --config config_v1.json --input_wavs_dir laughter/output --input_training_file laughter/output/training-ft.txt --input_validation_file laughter/output/validation_ft.txt --checkpoint_interval 5000 --training_epochs 50055
    ```

For other command line options, please refer to the training section.

## Inference
1. Make a `test_files` directory and copy the target laughter wav files into the directory.
2. Run the following command.
    ```
    python inference.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.

## Acknowledgements
I referred to [LaughNet](https://arxiv.org/abs/2110.04946), [HiFi-GAN](https://arxiv.org/abs/2010.05646),
and [vctk-silence-labels](https://github.com/nii-yamagishilab/vctk-silence-labels) to implement this.