A repository to train a mahjong agent who maybe can win a lot even when it's a "maybee".

### Usage
#### Prepare dataset
Visit [Tenhou record](https://tenhou.net/sc/raw/) to download paipu metadata. 

Use
```
cd scripts
python download_paipu.py
```
to download UML paipu based on your metadata.

Use 
```
cd scripts
python dataset_generator.py
```
to generate your dataset for supervised learning.

#### Supervised training
```
python train_supervised.py config/supervised.yaml [other arguments]
```

The configuration is implemented based on [utilsd](https://github.com/ultmaster/utilsd).
Remember to change the corresponding configurations (e.g. path to dataset) before training.
