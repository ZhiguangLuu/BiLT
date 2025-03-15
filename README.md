# Bidirectional Logits Tree: Pursuing Granularity Reconcilement in Fine-Grained Classification
This repository contains the source code of **Bidirectional Logits Tree: Pursuing Granularity Reconcilement in Fine-Grained Classification** at **AAAI 2025**.



## Requirement

```
torch 1.13.1+cu118
torchvision 0.14.1+cu118
```

## Data preparation
Refer to Repository:  [HAFrame](https://github.com/ltong1130ztr/HAFrame)

## Train

```bash
python main.py 	--start training \
				--arch custom_resnet50 \
				--batch-size 64 \
				--epochs 100 \
				--loss ours_l3 \
				--optimizer custom_sgd \
				--data fgvc-aircraft \
				--lr 0.1 \
				--output ckpt/fgvc/fgvc-aircraft-res50-sgd-bz64/ours-l3 \
				--weighting exp \
				--alpha 1.0 \
				--epsilon 0.5 \
				--gamma 1 \
				--beta 0.5 \
				--seed 0
```

## Test
```bash
python main.py 	--start testing \
				--arch custom_resnet50 \
				--batch-size 256 \
				--epochs 100 \
				--loss ours_l7 \
				--optimizer custom_sgd \
				--data inaturalist19-224 \
				--lr 0.1 \
				--output ckpt/iNat19/iNat-res50-sgd-bz256/ours-l7  \
				--weighting exp \
				--alpha 1.0 \
				--epsilon 0.5 \
				--gamma 1.5 \
				--beta 0.5 \
				--seed 0

```




## References and Acknowledgements
This repository is based on the following repos:
* [HAFrame](https://github.com/ltong1130ztr/HAFrame)
* [HAF](https://github.com/07Agarg/HAF)
* [CRM](https://github.com/sgk98/CRM-Better-Mistakes)
* [making-better-mistakes](https://github.com/fiveai/making-better-mistakes)
