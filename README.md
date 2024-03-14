

## **Environment**
- Linux
- Python 3.8
- Pytorch 1.9.1
- CUDA 11.1 (GPU with at least 11GB VRAM)

Other necessary packages:
```
pip install -r requirements.txt
```
- conda install ffmpeg
- pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh):
  -  sudo apt update
  -  sudo apt-get install libboost-dev
  -  git clone https://github.com/MPI-IS/mesh.git
  -  cd mesh
  -  BOOST_INCLUDE_DIRS=/path/to/boost/include make all
  -  make tests #用于测试是否成功

IMPORTANT: Please make sure to modify the `site-packages/torch/nn/modules/conv.py` file by commenting out the `self.padding_mode != 'zeros'` line to allow for replicated padding for ConvTranspose1d as shown [here](https://github.com/NVIDIA/tacotron2/issues/182).

## **Dataset Preparation**
### VOCASET
Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `vocaset/`. Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `vocaset/`. Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav`:
```
cd vocaset
python process_voca_data.py
```

### BIWI

Follow the [`BIWI/README.md`](BIWI/README.md) to preprocess BIWI dataset and put .npy/.wav files into `BIWI/vertices_npy` and `BIWI/wav`, and the `templates.pkl` into `BIWI/`.


## **Demo**
Download the pretrained models from [biwi_stage1.pth.tar](https://drive.google.com/file/d/1FSxey5Qug3MgAn69ymwFt8iuvwK6u37d/view?usp=sharing) & [biwi_stage2.pth.tar](https://drive.google.com/file/d/1gSNo9KYeIf6Mx3VYjRXQJBcg7Qv8UiUl/view?usp=sharing) Put the pretrained models under `BIWI`
[vocaset_stage1.pth.tar](https://drive.google.com/file/d/1RszIMsxcWX7WPlaODqJvax8M_dnCIzk5/view?usp=sharing) & [vocaset_stage2.pth.tar](https://drive.google.com/file/d/1phqJ_6AqTJmMdSq-__KY6eVwN4J9iCGP/view?usp=sharing). Put the pretrained models under `VOCASET` folders. Given the audio signal,

- to animate a mesh in FLAME topology, run: 
```
sh scripts/demo.sh vocaset
```
 可能需要开启一下代理`source /etc/network_turbo`，否则会出现connection error报错。如果出现了osmesa报错，则apt-get install -y python-opengl libosmesa6
 
 如果遇到 RuntimeError: The shape of the 3D attn_mask is torch.Size，是models.utils.py里的max_seq_len=600限制了最大序列，可以通过更改这个600到更大来测试性能。但是改了后需要自行训练，因为预训练模型是600

 改为60000后重新训练了一次，但是效果不好。而且依旧只能生成10s的视频
- to animate a mesh in BIWI topology, run: 
```
sh scripts/demo.sh BIWI
```
	This script will automatically generate the rendered videos in the `demo/output` folder. You can also put your own test audio file (.wav format) under the `demo/wav` folder and specify the arguments in `DEMO` section of `config/<dataset>/demo.yaml` accordingly (e.g., `demo_wav_path`, `condition`, `subject`, etc.).

## **Training / Testing**

The training/testing operation shares a similar command:
```
sh scripts/<train.sh|test.sh> <exp_name> config/<vocaset|BIWI>/<stage1|stage2>.yaml <vocaset|BIWI> <s1|s2>
```
Please replace `<exp_name>` with your own experiment name, `<vocaset|BIWI>` by the name of your target dataset, i.e., `vocaset` or `BIWI`. Change the `exp_dir` in both `scripts/train.sh` and `scripts/test.sh` if needed. We just take an example for default commands below.

### **Training for Discrete Motion Prior**

```
sh scripts/train.sh CodeTalker_s1 config/vocaset/stage1.yaml vocaset s1
```
如果在训练的时候报VQAutoEncoder error，可以参考（即在一阶段和二阶段使用不同的代码。）https://github.com/Doubiiu/CodeTalker/issues/5
二阶段训练时开一下代理

### **Training for Speech-Driven Motion Synthesis**
Make sure the paths of pre-trained models are correct, i.e., `vqvae_pretrained_path` and `wav2vec2model_path` in `config/<vocaset|BIWI>/stage2.yaml`.
```
sh scripts/train.sh CodeTalker_s2 config/vocaset/stage2.yaml vocaset s2
```
两阶段训练的模型都是在RUN文件夹中，一阶段训练完可直接训练二阶段。
### **Testing**
```
sh scripts/test.sh CodeTalker_s2 config/vocaset/stage2.yaml vocaset s2
```
这个testing好像不会给出结果，就是训练时的validation类似的，需要结果还是需要用demo来推理。用demo推理的时候改一下demo.yaml里的模型位置就行，用s2的模型。
## **Visualization with Audio**
Modify the paths in `scripts/render.sh` and run: 
```
sh scripts/render.sh
```

## **Evaluation on BIWI**
We provide the reference code for Lip Vertex Error & Upper-face Dynamics Deviation. Remember to change the paths in `scripts/cal_metric.sh`, and run:
```
sh scripts/cal_metric.sh
```
## **Play with Your Own Data**
###  Data Preparation

- Create the dataset directory `<dataset_dir>` in `CodeTalker` directory. 

- Place your vertices data (.npy files) and audio data (.wav files)  in `<dataset_dir>/vertices_npy` and `<dataset_dir>/wav` folders, respectively. 

- Save the templates of all subjects to a `templates.pkl` file and put it in `<dataset_dir>`, as done for BIWI and vocaset dataset. Export an arbitary template to .ply format and put it in `<dataset_dir>/`.

### Training, Testing & Visualization

- Create the corresponding config files in `config/<dataset_dir>` and modify the arguments in the config files.

- Check all the code segments releated to dataset information.

- Following the training/testing/visualization pipeline as done for BIWI and vocaset dataset.


