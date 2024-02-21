
## Train

python train.py --model="base" --cs_ratio=10 --batch_size=8 --blr=1e-4 --min_lr=1e-6 --epochs=200 --warmup_epochs=20 --train_folder="./data/train/" --val_folder="./data/val/" --image_size=132 --use_amp=True


Model will be saved in the saved_models folder, and log will be saved in the logs folder.

## Test

python test.py --model="base" --cs_ratio=10 --test_dataset="Set11" --is_save=True


The results will be saved as a results.csv file in the results folder, in the format of image name, PSNR, SSIM, and time.
