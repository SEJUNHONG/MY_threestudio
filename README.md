# NCthreestudio
The ImageDream, 3D Gaussian Splatting extension for threestudio. The original implementation can be found at https://github.com/bytedance/ImageDream and https://github.com/DSaurus/threestudio-3dgs. We thank them for their contribution to the 3D generation community. To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

# Docker
```
# docker <nvdiffrast Dockerfile>

docker pull harbor.sailio.ncsoft.com/sejun/nvdiffrast:latest
docker run -it --gpus all --shm-size=32g --name NC_3d --volume ./:/workspace harbor.sailio.ncsoft.com/sejun/nvdiffrast:latest
```

# Installation
```
pip install carvekit # Only used to create rgba images

# First install xformers (https://github.com/facebookresearch/xformers#installing-xformers)
# cuda 11.8 version
pip3 install -U xformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1 version
pip3 install -U xformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
git clone https://github.com/DSaurus/simple-knn.git
pip install ./diff-gaussian-rasterization
pip install ./simple-knn

# Then install other dependencies
pip install -r requirements.txt
pip uninstall transformer-engine
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```

# 

# Quick Start
```
# preprocess image (Create rgba images)
python rgba_image.py "extern/ImageDream/assets/corgi.png" --size 512 --border_ratio 0.0 # 
mkdir pretrained/omnidata
# cd load/omnidata
# assume gdown is installed
# gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
# gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt

# In addition to the rgba image, when creating depth and normal images together
# python preprocess_image.py "load/images/corgi.png" --size 512 --border_ratio 0.0  
```
### MVDream-NeRF

**Example running commands**

First train the coarse stage NeRF:

```sh
python launch.py --config configs/mvdream-nerf-coarse.yaml --train system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie"
```

Then convert the NeRF from the coarse stage to DMTet and train with differentiable rasterization:

```sh
python launch.py --config configs/mvdream-nerf-refine.yaml --train system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie" system.geometry_convert_from=outputs/mvdream-nerf-coarse/<tag>/ckpts/last.ckpt
```

### MVDream-3D Gaussian Splatting

**Example running commands**

First train the coarse stage NeRF:

```sh
python launch.py --config configs/mvdream-3dgs-coarse.yaml --train system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie"
```

### ImageDream-NeRF
Clone the modelcard on the [Huggingface ImageDream Model Page](https://huggingface.co/Peng-Wang/ImageDream/) under ```./extern/ImageDream/release_models/```

**Example running commands**

First train the coarse stage NeRF:

```sh
python launch.py --config configs/imagedream-nerf-coarse.yaml --train name="imagedream-nerf-coarse" tag="corgi" system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie" system.prompt_processor.image_path="extern/ImageDream/assets/corgi_rgba.png" system.guidance.ckpt_path="extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt" system.guidance.config_path="extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml" 
```

Then convert the NeRF from the coarse stage to DMTet and train with differentiable rasterization:

```sh
python launch.py --config configs/imagedream-nerf-refine.yaml --train system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie" system.prompt_processor.image_path="extern/ImageDream/assets/corgi_rgba.png" system.guidance.ckpt_path="extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt" system.guidance.config_path="extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml" system.geometry_convert_from=outputs/imagedream-nerf-coarse/<tag>/ckpts/last.ckpt
```

### ImageDream-3D Gaussian Splatting

**Example running commands**

First train the coarse stage NeRF:

```sh
python launch.py --config configs/imagedream-3dgs-coarse.yaml --train name="imagedream-3dgs-coarse" tag="corgi" system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie" system.prompt_processor.image_path="extern/ImageDream/assets/corgi_rgba.png" system.guidance.ckpt_path="extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt" system.guidance.config_path="extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml" 
```



# Resume from checkpoints
```
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train resume=path/to/trial/dir/ckpts/last.ckpt
# if the training has completed, you can still continue training for a longer time by setting trainer.max_steps
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train resume=path/to/trial/dir/ckpts/last.ckpt trainer.max_steps=20000
# you can also perform testing using resumed checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test resume=path/to/trial/dir/ckpts/last.ckpt
# note that the above commands use parsed configuration files from previous trials
# which will continue using the same trial directory
# if you want to save to a new trial directory, replace parsed.yaml with raw.yaml in the command

# only load weights from saved checkpoint but dont resume training (i.e. dont load optimizer state):
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train system.weights=path/to/trial/dir/ckpts/last.ckpt
```

### Export Meshes

To export the scene to texture meshes, use the `--export` option. We currently support exporting to obj+mtl, or obj with vertex colors.

```sh
# this uses default mesh-exporter configurations which exports obj+mtl
python launch.py --config outputs/<name>/<tag>/configs/parsed.yaml --export resume=outputs/<name>/<tag>/ckpts/last.ckpt system.exporter_type=mesh-exporter
# this uses gaussian mesh-exporter configurations which exports obj+mtl
python launch.py --config outputs/<name>/<tag>/configs/parsed.yaml --export resume=outputs/<name>/<tag>/ckpts/last.ckpt system.exporter_type=gaussian-mesh-exporter
# or
python launch.py --config path/to/config --export system.prompt_processor.prompt="a DSLR photo of a corgi taking a selfie" resume=outputs/<name>/<tag>/ckpts/last.ckpt
# specify system.exporter.fmt=obj to get obj with vertex colors
# you may also add system.exporter.save_uv=false to accelerate the process, suitable for a quick peek of the result
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj
# for NeRF-based methods (DreamFusion, Magic3D coarse, Latent-NeRF, SJC)
# you may need to adjust the isosurface threshold (25 by default) to get satisfying outputs
# decrease the threshold if the extracted model is incomplete, increase if it is extruded
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_threshold=10.
# use marching cubes of higher resolutions to get more detailed models
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256
```

# Citing

If you find MVDream, 3D Gaussian Splatting, and ImageDrema helpful, please consider citing:

```
@article{shi2023MVDream,
  author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
  title = {MVDream: Multi-view Diffusion for 3D Generation},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```

```
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```


``` bibtex
@article{wang2023imagedream,
  title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  author={Wang, Peng and Shi, Yichun},
  journal={arXiv preprint arXiv:2312.02201},
  year={2023}
}
```