## Backbone model: StarGAN v2
> **Paper**<br>
> StarGAN v2: Diverse Image Synthesis for Multiple Domains [link](https://arxiv.org/abs/1912.01865)<br>
> [Yunjey Choi](https://github.com/yunjey)\*, [Youngjung Uh](https://github.com/youngjung)\*, [Jaejun Yoo](http://jaejunyoo.blogspot.com/search/label/kr)\*, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<br>
> In CVPR 2020. (* indicates equal contribution)<br>

> **Official implementation in Pytorch**<br>
> The official Pytorch implementation of StarGAN v2 can be found at [clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)<br>

> **TensorFlow implementation**<br>
> The TensorFlow implementation of StarGAN v2 by our team member junho can be found at [clovaai/stargan-v2-tensorflow](https://github.com/clovaai/stargan-v2-tensorflow).

## Software installation
Clone this repository:

```bash
git clone https://github.com/KbeautyHair/KbeautyBaseline.git
cd KbeautyBaseline/
```

Install the dependencies:
```bash
conda create -n KbeautyBaseline python=3.6.7
conda activate KbeautyBaseline
```
* CUDA version 10.0
```bash
conda install -y pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.0 -c pytorch
```
* CUDA version 11.0
```bash
conda install -y pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=11.0 -c pytorch
```
```bash
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
```

## Datasets and pre-trained networks
We provide links to download K-hairstyle dataset we used to train the baseline model and the corresponding pre-trained networks. The datasets and checkpoints of the pre-trained networks are required to be downloaded and stored in the `data` and `expr/checkpoints` directories, respectively.

To download the K-hairstyle dataset, please visit this [dataset link](link) and for the pre-trained networks, visit this [checkpoint link](https://drive.google.com/file/d/1EYbJCUZBITAer2jscfguL3lNFZNMSaVy/view?usp=sharing), please.

## Translating images
After downloading the pre-trained networks, you can synthesize output images reflecting diverse hairstyles of reference images. The following commands will save generated images to the `expr/results` directory. 
To generate images, run the following command:
```bash
python main.py --mode sample --img_size 512 --num_domains 2 --resume_iter 60000 --w_hpf 0 \
               --checkpoint_dir expr/checkpoints/k-hairstyle --result_dir expr/results/k-hairstyle \
               --trg_domain [TARGET DOMAIN (e.g., 0)] --src_dir sample_images/src --ref_dir sample_images/ref               
```

## Evaluation metrics
To evaluate the baseline model using [Fr&eacute;chet Inception Distance (FID)](https://arxiv.org/abs/1706.08500), run the following commands:
```bash
python main.py --mode eval --img_size 512 --num_domains 2 --w_hpf 0 \
               --resume_iter 60000 --num_sample 1000 --val_batch_size 50 \        
               --train_img_dir data/mqset \
               --val_img_dir data/mqset \
               --checkpoint_dir expr/checkpoints/k-hairstyle \
               --eval_dir expr/eval/k-hairstyle --dataset_dir imagelists
```

## Training networks
To train the baseline model from scratch, run the following commands. Generated images and network checkpoints will be stored in the `expr/samples` and `expr/checkpoints` directories, respectively. Training takes about three days on a single Tesla V100 GPU. Please see [here](https://github.com/KbeautyHair/KbeautyBaseline/blob/master/main.py#L76-L122) for training arguments and a description of them.

```bash
python main.py --mode train --img_size 512 --num_domains 2 --w_hpf 0 \
               --lambda_reg 1 --lambda_sty 2 --lambda_ds 1 --lambda_cyc 2 \
               --batch_size 5 --val_batch_size 30 \
               --train_img_dir data/mqset --val_img_dir data/mqset --dataset_dir imagelists \
               --checkpoint_dir expr/checkpoints/k-hairstyle --eval_dir expr/eval/k-hairstyle --sample_dir expr/samples/k-hairstyle
```

## Reference
If you want to get more details of the original model Stargan v2, [this repository](https://github.com/clovaai/stargan-v2) will be useful.
The source code, pre-trained models of StarGAN v2 are available under [Creative Commons BY-NC 4.0](https://github.com/clovaai/stargan-v2/blob/master/LICENSE) license by NAVER Corporation. You can **use, copy, tranform and build upon** the material for **non-commercial purposes** as long as you give **appropriate credit** by citing our paper, and indicate if changes were made. 

## Contact
For business inquiries, please contact yuryueng@gmail.com or valther.ppk@gmail.com.<br/>	
For technical and other inquires, please contact codusl8@korea.ac.kr or specia1ktu@gmail.com.
