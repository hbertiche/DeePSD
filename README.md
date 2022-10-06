## DeePSD: Automatic deep skinning and pose space deformation for 3D garment animation

<img src="https://raw.githubusercontent.com/hbertiche/hbertiche.github.io/main/imgs/publications/DeePSD.png">

<a href="https://hbertiche.github.io/DeePSD">Project Page</a> | <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Bertiche_DeePSD_Automatic_Deep_Skinning_and_Pose_Space_Deformation_for_3D_ICCV_2021_paper.html">Paper</a> | <a href="https://arxiv.org/abs/2009.02715">arXiv</a>

## Abstract
>
>
>We present a novel solution to the garment animation problem through deep learning. Our contribution allows animating any template outfit with arbitrary topology and geometric complexity. Recent works develop models for garment edition, resizing and animation at the same time by leveraging the support body model (encoding garments as body homotopies). This leads to complex engineering solutions that suffer from scalability, applicability and compatibility. By limiting our scope to garment animation only, we are able to propose a simple model that can animate any outfit, independently of its topology, vertex order or connectivity. Our proposed architecture maps outfits to animated 3D models into the standard format for 3D animation (blend weights and blend shapes matrices), automatically providing of compatibility with any graphics engine. We also propose a methodology to complement supervised learning with an unsupervised physically based learning that implicitly solves collisions and enhances cloth quality.

<a href="mailto:hugo_bertiche@hotmail.com">Hugo Bertiche</a>, <a href="mailto:mmadadi@cvc.uab.cat">Meysam Madadi</a>, <a href="mailto:emilio.tyl@gmail.com">Emilio Tylson</a> and <a href="https://sergioescalera.com/">Sergio Escalera</a>

## Data
The dataset used on this work and this repository is <a href="http://chalearnlap.cvc.uab.es/dataset/38/description/">CLOTH3D</a>, with associated <a href="https://arxiv.org/abs/1912.02792">paper</a>.
<br>
Path to data has to be specified at 'values.py'. Note that it also asks for the path to preprocessings, described below.

### Preprocessing
In order to optimize data pipeline, we preprocess template outfits. The code to train the model assumes the preprocessing is done.
To perform this preprocessing, check the scripts at 'DeePSD/Preprocessing/'.
<ol>
  <li><b>outfit_verts.py</b> It creates 'txt' files for each sample. It relies on this for garment-to-outfit and outfit-to-garment conversions.</li>
  <li><b>rest.py</b> Rest garments are stored in OBJ files, which are encoded in ASCII. To increase efficiency, we convert this into binary format.</li>
  <li><b>faces2edges.py</b> Precomputes list of edges as [v0, v1] as int16 in binary format.</li>
  <li><b>laplacians.py</b> Precomputes laplacian matrices for each outfit.</li>
  <li><b>weights_prior.py</b> Precomputes blend weights labels by proximity to body in rest pose. Used for guiding learning in the first epoch.</li>
</ol>

## Train
Once all preprocessings have been completed. Just run 'train.py' and 'train_chi.py'.

## SMPL
We removed SMPL models in PKL format due to their size. The code will expect those as '/DeePSD/Model/smpl/model_f.pkl' and '/DeePSD/Model/smpl/model_m.pkl'.

## Citation
```
@inproceedings{bertiche2021deepsd,
  title={DeePSD: Automatic deep skinning and pose space deformation for 3D garment animation},
  author={Bertiche, Hugo and Madadi, Meysam and Tylson, Emilio and Escalera, Sergio},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5471--5480},
  year={2021}
}
```
