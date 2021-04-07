<h3>DATA</h3>
The dataset used on this work and this repository is <a href="http://chalearnlap.cvc.uab.es/dataset/38/description/">CLOTH3D</a>, with associated <a href="https://arxiv.org/abs/1912.02792">paper</a>.
<br>
Path to data has to be specified at 'values.py'. Note that it also asks for the path to preprocessings, described below.

<h4>PREPROCESSING</h4>
In order to optimize data pipeline, we preprocess template outfits. The code to train the model assumes the preprocessing is done.
To perform this preprocessing, check the scripts at 'DeePSD/Preprocessing/'.
<ol>
  <li><b>outfit_verts.py</b> It creates 'txt' files for each sample. It relies on this for garment-to-outfit and outfit-to-garment conversions.</li>
  <li><b>rest.py</b> Rest garments are stored in OBJ files, which are encoded in ASCII. To increase efficiency, we convert this into binary format.</li>
  <li><b>faces2edges.py</b> Precomputes list of edges as [v0, v1] as int16 in binary format.</li>
  <li><b>laplacians.py</b> Precomputes laplacian matrices for each outfit.</li>
  <li><b>weights_prior.py</b> Precomputes blend weights labels by proximity to body in rest pose. Used for guiding learning in the first epoch.</li>
</ol>

<h3>TRAIN</h3>
Once all preprocessings have been completed. Just run 'train.py' and 'train_chi.py'.

<h3>SMPL</h3>
We removed SMPL models in PKL format due to their size. The code will expect those as '/DeePSD/Model/smpl/model_f.pkl' and '/DeePSD/Model/smpl/model_m.pkl'.
